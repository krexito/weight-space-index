"""Zero-GPU Weight & Memory Streaming CDN (`wsi/cdn.py`).

Splits multi-entity `.moimem` memory banks into compact ~20 KB per-entity shards,
registers `.moipatch` behavioral patches in the same catalog, serves them over plain
HTTP without any GPU on the server, and streams + verifies (SHA-256) + attaches only
the matching payloads on demand onto a local compact model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import torch

from .moimem import MemoryBank, MemoryHandle, _cached_sha256, _sha256, find_layers, layer_down_proj
from .moipatch import MoIPatch, PatchHandle

INDEX_FILENAME = "moi_index.json"
CATALOG_FILENAME = "catalog.json"  # legacy fallback name


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "item"


def compute_model_down_proj_sha256(model, layer: int) -> str:
    """Compute cached base SHA-256 fingerprint of ``layer``'s MLP down-projection weight."""
    down = layer_down_proj(find_layers(model)[layer])
    return _cached_sha256(down.weight)


@dataclass
class CDNEntry:
    id: str
    kind: str  # "memory" | "patch"
    name: str
    keywords: list[str]
    blob_relpath: str
    size_bytes: int
    sha256: str
    layer: int | None = None
    model_sha256: str = ""
    base_model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CDNEntry":
        return cls(**d)


@dataclass
class StreamStats:
    matched_ids: list[str]
    kinds: list[str]
    bytes_downloaded: int
    cache_hits: int
    cache_misses: int
    resolve_ms: float
    download_ms: float
    attach_ms: float
    total_overhead_ms: float


class CDNCatalog:
    """Metadata index (``moi_index.json``) for `.moimem` memory shards and `.moipatch` payloads."""

    def __init__(
        self,
        root_dir: str | Path,
        entries: list[CDNEntry] | None = None,
        *,
        index_path: Path | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.blobs_dir = self.root_dir / "blobs"
        self.index_path: Path | None = index_path
        self.entries: list[CDNEntry] = list(entries or [])
        self._by_id: dict[str, CDNEntry] = {e.id: e for e in self.entries}

    def add(self, entry: CDNEntry) -> None:
        if entry.id in self._by_id:
            self.entries = [entry if e.id == entry.id else e for e in self.entries]
        else:
            self.entries.append(entry)
        self._by_id[entry.id] = entry

    def extend(self, new_entries: list[CDNEntry]) -> None:
        for e in new_entries:
            self.add(e)

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path is not None else (self.index_path or (self.root_dir / INDEX_FILENAME))
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "wsi-moi-index/1",
            "n_entries": len(self.entries),
            "total_blob_bytes": sum(e.size_bytes for e in self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out)
        self.index_path = out
        return out

    @classmethod
    def load(cls, root_or_catalog: str | Path) -> "CDNCatalog":
        p = Path(root_or_catalog)
        if p.is_dir():
            preferred = p / INDEX_FILENAME
            p = preferred if preferred.exists() else (p / CATALOG_FILENAME)
        data = json.loads(p.read_text(encoding="utf-8"))
        entries = [CDNEntry.from_dict(x) for x in data.get("entries", [])]
        return cls(p.parent, entries, index_path=p)

    def upsert_memory_shard(
        self,
        shard: MemoryBank,
        *,
        name: str | None = None,
        entry_id: str | None = None,
        dtype: str = "bf16",
        save_index: bool = True,
    ) -> CDNEntry:
        """Write/replace one single-entity `.moimem` shard and update the index atomically."""
        if len(shard) != 1:
            raise ValueError(f"upsert_memory_shard expects a 1-entity MemoryBank, got len={len(shard)}")
        raw_name = name or (shard.names[0] if shard.names else "entity")
        existing = next(
            (e for e in self.entries if e.kind == "memory" and e.name.lower() == raw_name.lower()),
            None,
        )
        eid = entry_id or (existing.id if existing is not None else f"mem_{_safe_slug(raw_name)}")
        rel = existing.blob_relpath if existing is not None else f"{eid}.moimem"
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        blob_path = self.blobs_dir / rel
        shard.save(blob_path, dtype=dtype)
        entry = CDNEntry(
            id=eid,
            kind="memory",
            name=raw_name,
            keywords=[raw_name.lower()],
            blob_relpath=rel,
            size_bytes=blob_path.stat().st_size,
            sha256=_file_sha256(blob_path),
            layer=int(shard.layer),
            model_sha256=str(shard.meta.get("down_proj_sha256", "")),
            base_model=str(shard.meta.get("base_model", "")),
        )
        self.add(entry)
        if save_index:
            self.save()
        return entry

    @staticmethod
    def _normalize_shas(model_sha256: str | set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
        if model_sha256 is None:
            return set()
        if isinstance(model_sha256, str):
            return {s.strip() for s in model_sha256.split(",") if s.strip()}
        return {str(s).strip() for s in model_sha256 if str(s).strip()}

    def resolve(
        self,
        query: str,
        *,
        model_sha256: str | set[str] | list[str] | tuple[str, ...] | None = None,
        kind: str | None = None,
        top_k: int = 4,
    ) -> list[CDNEntry]:
        q_norm = re.sub(r"\s+", " ", (query or "").strip().lower())
        if not q_norm:
            return []

        want_shas = self._normalize_shas(model_sha256)
        scored: list[tuple[float, int, CDNEntry]] = []
        for idx, entry in enumerate(self.entries):
            if kind and entry.kind != kind:
                continue
            if want_shas and entry.model_sha256 and entry.model_sha256 not in want_shas:
                continue

            best = 0.0
            phrases = entry.keywords or [entry.name.lower()]
            if entry.kind == "memory":
                for ph in phrases:
                    ph_clean = ph.strip().lower()
                    if ph_clean and re.search(rf"(?<!\w){re.escape(ph_clean)}(?!\w)", q_norm):
                        best = max(best, 100.0 + len(ph_clean))
            else:
                for ph in phrases:
                    ph_clean = ph.strip().lower()
                    if not ph_clean:
                        continue
                    if " " in ph_clean and ph_clean in q_norm:
                        best = max(best, 80.0 + len(ph_clean))
                    elif re.search(rf"(?<!\w){re.escape(ph_clean)}(?!\w)", q_norm):
                        best = max(best, 40.0 + len(ph_clean))

            if best > 0.0:
                scored.append((best, -idx, entry))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[: max(1, int(top_k))]]


def shard_memory_bank(
    bank_or_path: str | Path | MemoryBank,
    out_dir: str | Path,
    *,
    prefix: str = "mem",
    dtype: str = "bf16",
) -> list[CDNEntry]:
    """Split a multi-entity ``MemoryBank`` into ~20 KB single-entity `.moimem` CDN shards."""
    bank = bank_or_path if isinstance(bank_or_path, MemoryBank) else MemoryBank.load(bank_or_path)
    out_root = Path(out_dir)
    blobs_dir = out_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    model_sha = str(bank.meta.get("down_proj_sha256", ""))
    base_model = str(bank.meta.get("base_model", ""))
    facts = bank.meta.get("facts") or []

    entries: list[CDNEntry] = []
    n = len(bank)
    for i in range(n):
        raw_name = bank.names[i] if i < len(bank.names) else f"entity_{i}"
        entry_id = f"{prefix}_{i:05d}_{_safe_slug(raw_name)}"
        shard = MemoryBank(
            layer=bank.layer,
            c=bank.c[i : i + 1].clone(),
            r=bank.r[i : i + 1].clone(),
            ck=bank.ck[i : i + 1].clone(),
            thr=bank.thr,
            names=[raw_name],
            meta={
                "format": "moimem-shard/1",
                "method": bank.meta.get("method", "rome"),
                "base_model": base_model,
                "down_proj_sha256": model_sha,
                "down_proj_dtype": bank.meta.get("down_proj_dtype", "float32"),
                "fact": facts[i] if i < len(facts) else None,
            },
        )
        rel = f"{entry_id}.moimem"
        blob_path = blobs_dir / rel
        shard.save(blob_path, dtype=dtype)
        entries.append(
            CDNEntry(
                id=entry_id,
                kind="memory",
                name=raw_name,
                keywords=[raw_name.lower()],
                blob_relpath=rel,
                size_bytes=blob_path.stat().st_size,
                sha256=_file_sha256(blob_path),
                layer=int(bank.layer),
                model_sha256=model_sha,
                base_model=base_model,
            )
        )
    return entries


def register_moipatch_entry(
    patch_path: str | Path,
    out_dir: str | Path,
    *,
    keywords: list[str] | None = None,
    entry_id: str | None = None,
    model_sha256: str | None = None,
) -> CDNEntry:
    """Register a `.moipatch` file into a CDN catalog directory and return its entry."""
    src = Path(patch_path)
    patch = MoIPatch.load(src)
    out_root = Path(out_dir)
    blobs_dir = out_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    meta = patch.meta or {}
    name = str(meta.get("name") or src.stem)
    eid = entry_id or f"patch_{_safe_slug(name)}"
    rel = f"{eid}.moipatch"
    dst = blobs_dir / rel
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)

    if keywords is not None:
        kws = [str(k).strip().lower() for k in keywords if str(k).strip()]
    else:
        desc_tokens = re.findall(r"[a-z0-9_+-]{3,}", f"{name} {meta.get('description', '')}".lower())
        kws = sorted(set(desc_tokens))

    patch_sha = model_sha256 if model_sha256 is not None else (patch.edits[0].base_sha256 if patch.edits else "")
    return CDNEntry(
        id=eid,
        kind="patch",
        name=name,
        keywords=kws,
        blob_relpath=rel,
        size_bytes=dst.stat().st_size,
        sha256=_file_sha256(dst),
        layer=None,
        model_sha256=str(patch_sha),
        base_model=str(meta.get("base_model", "")),
        meta={"n_edits": len(patch.edits), "has_router": patch.router is not None},
    )


def build_cdn_catalog(
    out_dir: str | Path,
    *,
    memory_banks: list[str | Path | MemoryBank] | None = None,
    moipatches: list[tuple[str | Path, list[str] | None]] | None = None,
    shard_dtype: str = "bf16",
) -> CDNCatalog:
    """Build and persist a unified ``moi_index.json`` for `.moimem` shards and `.moipatch` files."""
    cat = CDNCatalog(out_dir)
    for b_idx, bank in enumerate(memory_banks or []):
        cat.extend(shard_memory_bank(bank, out_dir, prefix=f"mem{b_idx}", dtype=shard_dtype))
    for p_path, kws in (moipatches or []):
        cat.add(register_moipatch_entry(p_path, out_dir, keywords=kws))
    cat.save()
    return cat


class WeightCDNServer:
    """Zero-GPU threaded HTTP server serving `.moimem` shards and `.moipatch` files."""

    def __init__(
        self,
        catalog_or_dir: CDNCatalog | str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        sim_latency_ms: float = 0.0,
        sim_bandwidth_mbps: float = 0.0,
    ) -> None:
        self.catalog = (
            catalog_or_dir if isinstance(catalog_or_dir, CDNCatalog) else CDNCatalog.load(catalog_or_dir)
        )
        self.host = host
        self.requested_port = int(port)
        self.sim_latency_ms = float(sim_latency_ms)
        self.sim_bandwidth_mbps = float(sim_bandwidth_mbps)
        self._lock = threading.Lock()
        self._loaded_stat: tuple[int, int, int] | None = self._stat_key(self.catalog.index_path)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _stat_key(p: Path | None) -> tuple[int, int, int] | None:
        if p is None or not p.exists():
            return None
        st = p.stat()
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    @property
    def url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("CDN server is not running")
        h, p = self._httpd.server_address[:2]
        return f"http://{h}:{p}"

    def _get_fresh_catalog(self) -> CDNCatalog:
        idx_p = self.catalog.index_path
        if idx_p is not None and idx_p.exists():
            with self._lock:
                cur_stat = self._stat_key(idx_p)
                if cur_stat is not None and self._loaded_stat != cur_stat:
                    self.catalog = CDNCatalog.load(idx_p)
                    self._loaded_stat = cur_stat
        return self.catalog

    def start_background(self) -> "WeightCDNServer":
        if self._httpd is not None:
            return self
        srv_self = self
        blobs_root = self.catalog.blobs_dir.resolve()
        sim_lat = self.sim_latency_ms
        sim_bw = self.sim_bandwidth_mbps

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_json(self, code: int, obj: Any) -> None:
                raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/v1/health":
                    cat = srv_self._get_fresh_catalog()
                    kinds: dict[str, int] = {}
                    for e in cat.entries:
                        kinds[e.kind] = kinds.get(e.kind, 0) + 1
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "entries": len(cat.entries),
                            "total_blob_bytes": sum(e.size_bytes for e in cat.entries),
                            "kinds": kinds,
                            "sim_latency_ms": sim_lat,
                            "sim_bandwidth_mbps": sim_bw,
                        },
                    )
                    return

                if parsed.path == "/v1/resolve":
                    cat = srv_self._get_fresh_catalog()
                    qs = urllib.parse.parse_qs(parsed.query)
                    q = (qs.get("q") or [""])[0]
                    model_sha = (qs.get("model_sha") or [None])[0]
                    kind = (qs.get("kind") or [None])[0]
                    top_k = int((qs.get("top_k") or ["4"])[0])
                    hits = cat.resolve(q, model_sha256=model_sha, kind=kind, top_k=top_k)
                    matches = []
                    for e in hits:
                        d = e.to_dict()
                        d["blob_url"] = f"/v1/blobs/{urllib.parse.quote(e.blob_relpath)}"
                        matches.append(d)
                    self._send_json(200, {"query": q, "matches": matches})
                    return

                if parsed.path.startswith("/v1/blobs/"):
                    rel = urllib.parse.unquote(parsed.path[len("/v1/blobs/") :]).lstrip("/")
                    target = (blobs_root / rel).resolve()
                    if not target.is_relative_to(blobs_root) or not target.is_file():
                        self._send_json(404, {"error": "blob not found"})
                        return
                    raw = target.read_bytes()
                    delay_s = max(0.0, sim_lat / 1000.0)
                    if sim_bw > 0.0:
                        delay_s += len(raw) / (sim_bw * 125_000.0)
                    if delay_s > 0.0:
                        time.sleep(delay_s)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("X-WSI-SHA256", hashlib.sha256(raw).hexdigest())
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                self._send_json(404, {"error": "not found"})

        class _Server(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._httpd = _Server((self.host, self.requested_port), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "WeightCDNServer":
        return self.start_background()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


class WeightStreamClient:
    """Streaming client supporting both API servers (`mode='api'`) and plain static hosts (`mode='static'`)."""

    def __init__(
        self,
        base_url: str,
        *,
        mode: str = "api",
        index_filename: str = INDEX_FILENAME,
        max_entries: int = 64,
        timeout_s: float = 10.0,
    ) -> None:
        if mode not in ("api", "static"):
            raise ValueError(f"Unsupported mode {mode!r}; expected 'api' or 'static'")
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.index_filename = index_filename.lstrip("/")
        self.max_entries = max(1, int(max_entries))
        self.timeout_s = float(timeout_s)
        self._cache: OrderedDict[tuple[str, str], tuple[str, Any]] = OrderedDict()
        self._static_catalog: CDNCatalog | None = None

    def clear_cache(self) -> None:
        self._cache.clear()

    def refresh_index(self) -> CDNCatalog:
        """Fetch and parse ``moi_index.json`` from a static CDN origin (GitHub Raw, R2, S3)."""
        data = self._get_json("/" + self.index_filename)
        entries = [CDNEntry.from_dict(x) for x in data.get("entries", [])]
        self._static_catalog = CDNCatalog(self.base_url, entries)
        return self._static_catalog

    def _get_json(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_bytes(self, blob_url: str) -> bytes:
        full = blob_url if blob_url.startswith("http") else (self.base_url + blob_url)
        req = urllib.request.Request(full, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read()

    @staticmethod
    def _parse_payload(kind: str, raw: bytes) -> Any:
        suffix = ".moimem" if kind == "memory" else ".moipatch"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(raw)
            tmp_path = Path(tf.name)
        try:
            return MemoryBank.load(tmp_path) if kind == "memory" else MoIPatch.load(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _fetch_entry(self, entry: dict[str, Any], *, verify_sha256: bool) -> tuple[Any, int, bool]:
        key = (str(entry["id"]), str(entry["sha256"]))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][1], 0, True

        raw = self._get_bytes(str(entry["blob_url"]))
        if verify_sha256 and entry.get("sha256"):
            got = hashlib.sha256(raw).hexdigest()
            want = str(entry["sha256"])
            if got != want:
                raise ValueError(
                    f"SHA-256 mismatch for CDN entry {entry['id']!r}: expected {want}, got {got}"
                )
        obj = self._parse_payload(str(entry["kind"]), raw)
        self._cache[key] = (str(entry["kind"]), obj)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return obj, len(raw), False

    @contextmanager
    def stream_for(
        self,
        prompt: str,
        model,
        *,
        model_sha256: str | set[str] | list[str] | tuple[str, ...] | None = None,
        kind: str | None = None,
        top_k: int = 4,
        gated: bool = True,
        verify_sha256: bool = True,
        verify_model_fingerprint: bool = True,
    ) -> Iterator[StreamStats]:
        """Resolve, download (or hit cache), attach matched `.moimem`/`.moipatch` to `model`, and restore on exit."""
        t_start = time.perf_counter()
        t_res0 = time.perf_counter()
        if self.mode == "static":
            if self._static_catalog is None:
                self.refresh_index()
            assert self._static_catalog is not None
            hits = self._static_catalog.resolve(prompt, model_sha256=model_sha256, kind=kind, top_k=top_k)
            matches = []
            for e in hits:
                d = e.to_dict()
                d["blob_url"] = f"/blobs/{urllib.parse.quote(e.blob_relpath)}"
                matches.append(d)
        else:
            params: dict[str, Any] = {"q": prompt, "top_k": int(top_k)}
            shas = CDNCatalog._normalize_shas(model_sha256)
            if shas:
                params["model_sha"] = ",".join(sorted(shas))
            if kind:
                params["kind"] = kind
            resolved = self._get_json("/v1/resolve?" + urllib.parse.urlencode(params))
            matches = list(resolved.get("matches") or [])
        resolve_ms = (time.perf_counter() - t_res0) * 1000.0

        bytes_downloaded = 0
        cache_hits = 0
        cache_misses = 0
        mem_by_layer: dict[int, list[MemoryBank]] = defaultdict(list)
        patches: list[MoIPatch] = []

        t_dl0 = time.perf_counter()
        for entry in matches:
            obj, n_bytes, hit = self._fetch_entry(entry, verify_sha256=verify_sha256)
            bytes_downloaded += n_bytes
            if hit:
                cache_hits += 1
            else:
                cache_misses += 1
            if entry["kind"] == "memory":
                mem_by_layer[int(obj.layer)].append(obj)
            else:
                patches.append(obj)
        download_ms = (time.perf_counter() - t_dl0) * 1000.0 if matches else 0.0

        active_handles: list[MemoryHandle | PatchHandle] = []
        t_att0 = time.perf_counter()
        try:
            for layer_idx, banks in mem_by_layer.items():
                if len(banks) == 1:
                    merged = banks[0]
                else:
                    b0 = banks[0]
                    merged = MemoryBank(
                        layer=layer_idx,
                        c=torch.cat([b.c for b in banks], dim=0),
                        r=torch.cat([b.r for b in banks], dim=0),
                        ck=torch.cat([b.ck for b in banks], dim=0),
                        thr=b0.thr,
                        names=[n for b in banks for n in b.names],
                        meta=dict(b0.meta),
                    )
                if not verify_model_fingerprint and "down_proj_sha256" in merged.meta:
                    merged = MemoryBank(
                        layer=merged.layer,
                        c=merged.c,
                        r=merged.r,
                        ck=merged.ck,
                        thr=merged.thr,
                        names=merged.names,
                        meta={k: v for k, v in merged.meta.items() if k != "down_proj_sha256"},
                    )
                mh = merged.attach(model).on(gated=gated)
                active_handles.append(mh)

            for patch in patches:
                ph = patch.attach(model, verify_fingerprint=verify_model_fingerprint)
                ph.on()
                active_handles.append(ph)

            attach_ms = (time.perf_counter() - t_att0) * 1000.0 if matches else 0.0
            total_overhead_ms = (time.perf_counter() - t_start) * 1000.0
            yield StreamStats(
                matched_ids=[str(m["id"]) for m in matches],
                kinds=[str(m["kind"]) for m in matches],
                bytes_downloaded=bytes_downloaded,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                resolve_ms=round(resolve_ms, 3),
                download_ms=round(download_ms, 3),
                attach_ms=round(attach_ms, 3),
                total_overhead_ms=round(total_overhead_ms, 3),
            )
        finally:
            for h in reversed(active_handles):
                h.off()
