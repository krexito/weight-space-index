"""Bake (export) `.moipatch` edits into checkpoints and serve base models with live hot-swapping.

Two zero-dependency workflows for end users and apps:
  1. ``bake_moipatch`` / ``python -m wsi.moipatch export``: stream-patch `.moipatch` edits directly
     into safetensors shards on disk (ready for HuggingFace, vLLM, or GGUF conversion).
  2. ``PatchRegistry`` + ``create_hotswap_server`` / ``python -m wsi.moipatch serve``: keep one
     base model loaded and hot-swap, compose, load, or unload `.moipatch` files in milliseconds
     behind an OpenAI-compatible ``/v1/chat/completions`` API, WSI hot-swap REST API, and local web chat UI.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from .moipatch import _DTYPES, Edit, MoIPatch, _sha256, apply_edit, module_suffix

_SKIP_SAFETENSORS = {"delta_lowrank.safetensors"}


def _atomic_write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "auto":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _index_checkpoint_shards(model_dir: Path) -> tuple[list[Path], dict[str, list[tuple[str, str]]]]:
    """Map ``module_suffix -> [(shard_rel_name, tensor_key), ...]`` without loading tensors."""
    from safetensors import safe_open

    idx_file = model_dir / "model.safetensors.index.json"
    suffix_map: dict[str, list[tuple[str, str]]] = {}
    shard_names: set[str] = set()

    if idx_file.exists():
        wmap = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
        for k, fname in wmap.items():
            shard_names.add(str(fname))
            if k.endswith(".weight") and "mtp" not in k and "visual" not in k:
                try:
                    suffix_map.setdefault(module_suffix(k[: -len(".weight")]), []).append((str(fname), k))
                except ValueError:
                    pass
        shards = [model_dir / s for s in sorted(shard_names)]
    else:
        shards = [
            f
            for f in sorted(model_dir.glob("*.safetensors"))
            if not f.name.startswith("edit_") and f.name not in _SKIP_SAFETENSORS
        ]
        for shard_p in shards:
            with safe_open(str(shard_p), "pt") as sf:
                for k in sf.keys():
                    if k.endswith(".weight") and "mtp" not in k and "visual" not in k:
                        try:
                            suffix_map.setdefault(module_suffix(k[: -len(".weight")]), []).append((shard_p.name, k))
                        except ValueError:
                            pass

    if not shards:
        raise FileNotFoundError(f"No safetensors checkpoint files found in {model_dir}")
    return shards, suffix_map


def bake_moipatch(
    model_dir: str | Path,
    patches: MoIPatch | str | Path | Sequence[MoIPatch | str | Path],
    out_dir: str | Path,
    *,
    verify_fingerprint: bool = True,
    allow_routed: bool = False,
    copy_untouched: str = "auto",
) -> dict[str, Any]:
    """Stream-bake one or more `.moipatch` files directly into a standalone HuggingFace checkpoint directory."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if copy_untouched not in ("auto", "copy"):
        raise ValueError(f"copy_untouched must be 'auto' or 'copy', got {copy_untouched!r}")

    mdir = Path(model_dir)
    odir = Path(out_dir)
    if not mdir.exists():
        raise FileNotFoundError(f"Base model directory not found: {mdir}")
    if odir.resolve() == mdir.resolve():
        raise ValueError("out_dir must differ from model_dir to avoid overwriting the base checkpoint")

    raw_list: Sequence[MoIPatch | str | Path] = (
        [patches] if isinstance(patches, (MoIPatch, str, Path)) else list(patches)
    )
    if not raw_list:
        raise ValueError("At least one patch is required for baking")

    patch_items: list[tuple[str, MoIPatch]] = []
    for i, item in enumerate(raw_list):
        if isinstance(item, MoIPatch):
            lbl = str(item.meta.get("name") or f"patch_{i}")
            patch_items.append((lbl, item))
        else:
            p_path = Path(item)
            patch_items.append((p_path.stem, MoIPatch.load(p_path)))

    routed_names = [lbl for lbl, p in patch_items if p.router is not None]
    if routed_names and not allow_routed:
        raise ValueError(
            f"Patch(es) {routed_names} contain a runtime router/safety veto (`router`). "
            "Baking writes static weights (`always_on`) without runtime routing. "
            "Pass allow_routed=True (or --force) to bake statically anyway, or use `wsi.moipatch serve` for live routed hot-swap."
        )

    shards, suffix_map = _index_checkpoint_shards(mdir)
    shard_edits: dict[str, list[tuple[str, str, Edit]]] = {}
    for lbl, p in patch_items:
        for e in p.edits:
            hits = suffix_map.get(e.module, [])
            if len(hits) != 1:
                raise KeyError(f"{e.module}: expected 1 matching weight tensor in {mdir}, found {len(hits)}")
            shard_rel, tensor_key = hits[0]
            shard_edits.setdefault(shard_rel, []).append((lbl, tensor_key, e))

    odir.mkdir(parents=True, exist_ok=True)
    touched_names = set(shard_edits.keys())
    shard_set = {s.name for s in shards}

    # Copy configs, tokenizers, index JSON, and link/copy untouched shards
    for entry in sorted(mdir.iterdir()):
        if entry.name in (".git", "__pycache__"):
            continue
        dst = odir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst, dirs_exist_ok=True)
        elif entry.name in touched_names:
            continue
        elif entry.name in shard_set:
            _link_or_copy(entry, dst, copy_untouched)
        else:
            shutil.copy2(entry, dst)

    baked_modules: dict[str, dict[str, Any]] = {}
    verified_keys: set[str] = set()
    for shard_rel, ops in shard_edits.items():
        src_shard = mdir / shard_rel
        dst_shard = odir / shard_rel
        with safe_open(str(src_shard), "pt") as sf:
            raw_meta = sf.metadata() or {}
            metadata = {str(k): str(v) for k, v in raw_meta.items()}
            # Clone before closing mmap so tensor buffers never dangle on Windows
            tensors = {k: sf.get_tensor(k).clone() for k in sf.keys()}

        for lbl, tensor_key, e in ops:
            w_cur = tensors[tensor_key]
            if tensor_key not in verified_keys:
                fp_dtype = _DTYPES.get(e.base_dtype, w_cur.dtype)
                orig_sha = _sha256(w_cur.detach().to(fp_dtype))
                if verify_fingerprint and e.base_sha256 and orig_sha != e.base_sha256:
                    raise ValueError(
                        f"{e.module} in {shard_rel}: base SHA-256 mismatch ({orig_sha[:12]} != {e.base_sha256[:12]})"
                    )
                verified_keys.add(tensor_key)
                baked_modules[e.module] = {
                    "shard": shard_rel,
                    "tensor_key": tensor_key,
                    "base_sha256": orig_sha,
                    "patches": [],
                }
            tensors[tensor_key] = apply_edit(w_cur, e.u, e.idx, e.alpha, e.v)
            baked_modules[e.module]["patches"].append({"patch": lbl, "alpha": float(e.alpha)})
            baked_modules[e.module]["baked_sha256"] = _sha256(tensors[tensor_key])

        dst_shard.parent.mkdir(parents=True, exist_ok=True)
        if dst_shard.exists() or dst_shard.is_symlink():
            dst_shard.unlink()
        tmp_shard = dst_shard.with_name(dst_shard.name + ".tmp")
        save_file(tensors, str(tmp_shard), metadata=metadata)
        os.replace(tmp_shard, dst_shard)

    provenance = {
        "base_model_dir": str(mdir),
        "out_dir": str(odir),
        "patches": [{"label": lbl, "name": p.meta.get("name", lbl), "has_router": p.router is not None} for lbl, p in patch_items],
        "had_router_warning": bool(routed_names),
        "routed_patches": routed_names,
        "n_total_shards": len(shards),
        "n_touched_shards": len(touched_names),
        "modules": baked_modules,
    }
    _atomic_write_bytes(odir / "wsi_baked_patches.json", json.dumps(provenance, indent=2).encode("utf-8"))
    return provenance


def _format_chat_messages(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    norm_msgs = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in messages]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                norm_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(norm_msgs, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in norm_msgs) + "\nASSISTANT:"


def _resolve_pad_eos_ids(model: Any, tokenizer: Any) -> tuple[int | None, int | None]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_id = eos_id[0] if eos_id else None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_id, list):
        pad_id = pad_id[0] if pad_id else None
    if pad_id is None:
        pad_id = eos_id
    return (int(eos_id) if eos_id is not None else None), (int(pad_id) if pad_id is not None else None)


class PatchRegistry:
    """Thread-safe registry for loading, composing, and hot-swapping `.moipatch` handles on one model."""

    def __init__(self, model: Any, tokenizer: Any, *, verify_fingerprint: bool = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verify_fingerprint = bool(verify_fingerprint)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._active_order: list[str] = []

    def _model_device(self) -> torch.device:
        for p in self.model.parameters():
            return p.device
        return torch.device("cpu")

    def _entry_summary(self, name: str, entry: dict[str, Any]) -> dict[str, Any]:
        patch: MoIPatch = entry["patch"]
        handle = entry["handle"]
        return {
            "name": name,
            "patch_name": str(patch.meta.get("name") or name),
            "description": str(patch.meta.get("description") or ""),
            "path": entry["path"],
            "mode": entry["mode"],
            "active": name in self._active_order and handle is not None,
            "has_router": patch.router is not None,
            "has_veto": bool(patch.router and patch.router.get("veto_dir") is not None),
            "router_layer": patch.router.get("layer") if patch.router else None,
            "n_edits": len(patch.edits),
            "edited_modules": [e.module for e in patch.edits],
            "stats": handle.stats if handle is not None else None,
        }

    def load_patch(
        self,
        name: str,
        patch_or_path: MoIPatch | str | Path,
        *,
        mode: str = "auto",
        activate: bool = False,
    ) -> dict[str, Any]:
        """Validate base fingerprint and register a `.moipatch` under `name`, optionally activating it."""
        clean_name = str(name).strip()
        if not clean_name or clean_name in ("base", "auto"):
            raise ValueError(f"Invalid patch name {name!r} (reserved: 'base', 'auto')")
        if mode not in ("auto", "always_on", "off"):
            raise ValueError(f"Unsupported mode {mode!r}; expected 'auto', 'always_on', or 'off'")

        if isinstance(patch_or_path, MoIPatch):
            patch = patch_or_path
            src_path = None
        else:
            src_path = str(patch_or_path)
            patch = MoIPatch.load(src_path)

        # Cryptographic verification before registration without keeping handle active
        chk = patch.attach(self.model, verify_fingerprint=self.verify_fingerprint)
        if not chk.fingerprint_ok():
            raise ValueError(f"Fingerprint verification failed for patch {clean_name!r}")

        with self._lock:
            old = self._entries.get(clean_name)
            if old is not None and old["handle"] is not None:
                old["handle"].close()
                self._active_order = [n for n in self._active_order if n != clean_name]

            self._entries[clean_name] = {
                "patch": patch,
                "path": src_path,
                "mode": mode,
                "handle": None,
            }
            if activate:
                new_active = [*self._active_order, clean_name]
                self._apply_active_locked(new_active, mode=mode)
            return self._entry_summary(clean_name, self._entries[clean_name])

    def _apply_active_locked(self, names: Sequence[str], *, mode: str | None = None) -> list[dict[str, Any]]:
        target_names: list[str] = []
        for raw_n in names:
            nm = str(raw_n).strip()
            if not nm:
                continue
            if nm not in self._entries:
                raise KeyError(f"Unknown patch {nm!r}; loaded patches: {sorted(self._entries)}")
            if nm not in target_names:
                target_names.append(nm)

        if mode is not None and mode not in ("auto", "always_on", "off"):
            raise ValueError(f"Unsupported mode {mode!r}; expected 'auto', 'always_on', or 'off'")

        target_set = set(target_names)
        # Rebuild active stack if order changed or any active patch needs a mode switch
        needs_rebuild = target_names != self._active_order
        if not needs_rebuild and mode is not None:
            needs_rebuild = any(self._entries[nm]["mode"] != mode for nm in target_names)

        if needs_rebuild:
            for nm in reversed(self._active_order):
                ent = self._entries.get(nm)
                if ent is not None and ent["handle"] is not None:
                    ent["handle"].close()
                    ent["handle"] = None

            self._active_order = []
            for nm in target_names:
                ent = self._entries[nm]
                eff_mode = mode if mode is not None else ent["mode"]
                ent["mode"] = eff_mode
                ent["handle"] = ent["patch"].enable_routed(
                    self.model, mode=eff_mode, verify_fingerprint=self.verify_fingerprint
                )
                self._active_order.append(nm)

        # Clean up any handle outside target_set just in case
        for nm, ent in self._entries.items():
            if nm not in target_set and ent["handle"] is not None:
                ent["handle"].close()
                ent["handle"] = None

        return [self._entry_summary(nm, self._entries[nm]) for nm in self._active_order]

    def activate(self, names: Sequence[str], *, mode: str | None = None) -> list[dict[str, Any]]:
        """Hot-swap the active patch stack in milliseconds without reloading model weights."""
        with self._lock:
            return self._apply_active_locked(names, mode=mode)

    def unload_patch(self, name: str) -> bool:
        """Deactivate (if active) and remove a patch from the registry."""
        with self._lock:
            ent = self._entries.pop(name, None)
            if ent is None:
                return False
            if ent["handle"] is not None:
                ent["handle"].close()
            self._active_order = [n for n in self._active_order if n != name]
            return True

    def list_patches(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._entry_summary(nm, ent) for nm, ent in self._entries.items()]

    def active_names(self) -> list[str]:
        with self._lock:
            return list(self._active_order)

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model_name: str | None = None,
        patches: Sequence[str] | None = None,
        mode: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Generate a chat completion under either the live active stack or a per-request hot-swap."""
        with self._lock:
            prev_order = list(self._active_order)
            prev_modes = {nm: self._entries[nm]["mode"] for nm in prev_order}

            temp_swap = False
            if patches is not None:
                req_names = list(patches)
                temp_swap = True
            elif model_name == "base":
                req_names = []
                temp_swap = True
            elif model_name is not None and model_name in self._entries:
                req_names = [model_name]
                temp_swap = True
            else:
                req_names = list(prev_order)

            try:
                if temp_swap or mode is not None:
                    self._apply_active_locked(req_names, mode=mode)

                active_now = list(self._active_order)
                for nm in active_now:
                    h = self._entries[nm]["handle"]
                    if h is not None:
                        h.reset_stats()

                prompt_str = _format_chat_messages(self.tokenizer, messages)
                eos_id, pad_id = _resolve_pad_eos_ids(self.model, self.tokenizer)
                p_ids = (
                    self.tokenizer.encode(prompt_str, add_special_tokens=False)
                    if hasattr(self.tokenizer, "encode")
                    else []
                )
                if not p_ids:
                    p_ids = [eos_id or 1]

                dev = self._model_device()
                inp = torch.tensor([p_ids], dtype=torch.long, device=dev)
                gen_kw: dict[str, Any] = {"max_new_tokens": max(1, int(max_tokens))}
                if float(temperature) > 0.0:
                    gen_kw["do_sample"] = True
                    gen_kw["temperature"] = float(temperature)
                if pad_id is not None:
                    gen_kw["pad_token_id"] = pad_id

                with torch.no_grad():
                    out_ids = self.model.generate(inp, **gen_kw)

                gen_ids = [int(t) for t in out_ids[0, len(p_ids) :].tolist()]
                if eos_id is not None and eos_id in gen_ids:
                    gen_ids = gen_ids[: gen_ids.index(eos_id)]
                text = (
                    self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    if hasattr(self.tokenizer, "decode")
                    else " ".join(map(str, gen_ids))
                )
                telemetry = {
                    nm: self._entries[nm]["handle"].stats
                    for nm in active_now
                    if self._entries[nm]["handle"] is not None
                }
                return {
                    "text": text,
                    "prompt_tokens": len(p_ids),
                    "completion_tokens": len(gen_ids),
                    "active_patches": active_now,
                    "wsi_telemetry": telemetry,
                }
            finally:
                if temp_swap or mode is not None:
                    for nm in reversed(self._active_order):
                        ent = self._entries.get(nm)
                        if ent is not None and ent["handle"] is not None:
                            ent["handle"].close()
                            ent["handle"] = None
                    self._active_order = []
                    for nm in prev_order:
                        if nm in self._entries:
                            ent = self._entries[nm]
                            ent["mode"] = prev_modes[nm]
                            ent["handle"] = ent["patch"].enable_routed(
                                self.model, mode=prev_modes[nm], verify_fingerprint=self.verify_fingerprint
                            )
                            self._active_order.append(nm)


_WEB_CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WSI Live Hot-Swap Chat</title>
<style>
  :root { color-scheme: dark; --bg: #0f1117; --card: #181c26; --border: #2a3040; --accent: #4f8cff; --text: #e6eaf2; --muted: #9aa4b8; --ok: #34d399; }
  * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
  aside { background: var(--card); border-right: 1px solid var(--border); padding: 16px; display: flex; flex-direction: column; gap: 14px; overflow-y: auto; }
  main { display: flex; flex-direction: column; height: 100vh; }
  h1 { font-size: 16px; margin: 0; letter-spacing: .2px; }
  h2 { font-size: 12px; text-transform: uppercase; color: var(--muted); margin: 0 0 8px 0; }
  .box { border: 1px solid var(--border); border-radius: 10px; padding: 12px; background: #131720; }
  .patch-row { display: flex; align-items: flex-start; gap: 8px; padding: 6px 0; border-bottom: 1px solid #1f2533; font-size: 13px; }
  .patch-row:last-child { border-bottom: 0; }
  .tag { font-size: 11px; padding: 2px 6px; border-radius: 999px; background: #232a3b; color: var(--muted); }
  .tag.on { background: #163229; color: var(--ok); }
  label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }
  input, select, button, textarea { width: 100%; border-radius: 8px; border: 1px solid var(--border); background: #0f1117; color: var(--text); padding: 8px 10px; font-size: 13px; }
  button { background: var(--accent); border-color: var(--accent); font-weight: 600; cursor: pointer; }
  button.sec { background: #232a3b; border-color: var(--border); }
  #chat { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 780px; padding: 12px 14px; border-radius: 12px; line-height: 1.45; white-space: pre-wrap; font-size: 14px; }
  .msg.user { align-self: flex-end; background: #1e325c; }
  .msg.assistant { align-self: flex-start; background: var(--card); border: 1px solid var(--border); }
  .meta { font-size: 11px; color: var(--muted); margin-top: 6px; }
  form#send { display: flex; gap: 8px; padding: 14px 20px; border-top: 1px solid var(--border); background: var(--card); }
  form#send textarea { resize: none; height: 46px; flex: 1; }
  form#send button { width: 110px; }
</style>
</head>
<body>
<aside>
  <div>
    <h1>WSI Hot-Swap Studio</h1>
    <div id="status" class="meta">Connecting...</div>
  </div>
  <div class="box">
    <h2>Active Patches (Live Swap)</h2>
    <div id="patch-list">No patches loaded</div>
    <div style="margin-top:10px">
      <label for="mode-sel">Routing Mode</label>
      <select id="mode-sel">
        <option value="auto">auto (routed + safety veto)</option>
        <option value="always_on">always_on</option>
        <option value="off">off (base model)</option>
      </select>
    </div>
    <button class="sec" style="margin-top:8px" onclick="applySwap()">Apply Hot-Swap</button>
  </div>
  <div class="box">
    <h2>Load .moipatch From Path</h2>
    <label for="load-name">Name</label>
    <input id="load-name" placeholder="e.g. abliterated-sfw">
    <label for="load-path" style="margin-top:6px">File path</label>
    <input id="load-path" placeholder="/path/to/patch.moipatch">
    <button class="sec" style="margin-top:8px" onclick="loadPatch()">Load &amp; Activate</button>
  </div>
  <button class="sec" onclick="clearChat()">Clear Conversation</button>
</aside>
<main>
  <div id="chat"></div>
  <form id="send" onsubmit="sendMsg(event)">
    <textarea id="prompt" placeholder="Type a message (Shift+Enter for newline, Enter to send)..."></textarea>
    <button type="submit">Send</button>
  </form>
</main>
<script>
const historyMsgs = [];
async function refreshPatches() {
  const res = await fetch('/v1/wsi/patches');
  const data = await res.json();
  const root = document.getElementById('patch-list');
  document.getElementById('status').textContent = 'Active: ' + (data.active.length ? data.active.join(', ') : 'base (no patches)');
  if (!data.patches.length) { root.textContent = 'No patches loaded (using base model)'; return; }
  root.innerHTML = '';
  for (const p of data.patches) {
    const row = document.createElement('div');
    row.className = 'patch-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.style.width = '16px'; cb.checked = p.active; cb.dataset.name = p.name;
    cb.onchange = applySwap;
    const info = document.createElement('div');
    info.innerHTML = `<strong>${p.name}</strong> <span class="tag ${p.active?'on':''}">${p.has_veto?'vetoed':(p.has_router?'routed':'static')}</span><div class="meta">${p.edited_modules.join(', ')}</div>`;
    row.appendChild(cb); row.appendChild(info); root.appendChild(row);
  }
}
async function applySwap() {
  const names = [...document.querySelectorAll('#patch-list input[type=checkbox]:checked')].map(x => x.dataset.name);
  const mode = document.getElementById('mode-sel').value;
  await fetch('/v1/wsi/patches/activate', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({patches: names, mode})
  });
  await refreshPatches();
}
async function loadPatch() {
  const name = document.getElementById('load-name').value.trim();
  const path = document.getElementById('load-path').value.trim();
  if (!name || !path) return;
  const r = await fetch('/v1/wsi/patches/load', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({name, path, activate: true, mode: document.getElementById('mode-sel').value})
  });
  const d = await r.json();
  if (!r.ok) { alert(d.error || 'Failed to load patch'); return; }
  await refreshPatches();
}
function addBubble(role, text, meta='') {
  const d = document.createElement('div'); d.className = 'msg ' + role; d.textContent = text;
  if (meta) { const m = document.createElement('div'); m.className = 'meta'; m.textContent = meta; d.appendChild(m); }
  const chat = document.getElementById('chat'); chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
}
function clearChat() { historyMsgs.length = 0; document.getElementById('chat').innerHTML = ''; }
async function sendMsg(ev) {
  ev.preventDefault();
  const ta = document.getElementById('prompt'); const txt = ta.value.trim();
  if (!txt) return;
  ta.value = ''; historyMsgs.push({role: 'user', content: txt}); addBubble('user', txt);
  const res = await fetch('/v1/chat/completions', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({model: 'auto', messages: historyMsgs, max_tokens: 256, temperature: 0.0})
  });
  const data = await res.json();
  if (!res.ok) { addBubble('assistant', 'Error: ' + (data.error || res.statusText)); return; }
  const reply = data.choices[0].message.content;
  historyMsgs.push({role: 'assistant', content: reply});
  const w = data.wsi || {};
  const fired = Object.entries(w.telemetry || {}).map(([k, s]) => `${k}:fired=${s.fired}/${s.calls}`).join(' | ');
  addBubble('assistant', reply, `active=[${(w.active_patches || []).join(', ') || 'base'}]` + (fired ? ` | ${fired}` : ''));
  await refreshPatches();
}
document.getElementById('prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); document.getElementById('send').requestSubmit(); }
});
refreshPatches();
</script>
</body>
</html>
"""


def dispatch_hotswap_request(
    registry: PatchRegistry,
    method: str,
    raw_path: str,
    raw_body: bytes = b"",
    *,
    base_model_name: str = "base",
) -> tuple[int, str, bytes]:
    """Pure HTTP router returning ``(status_code, content_type, body_bytes)`` for all hot-swap endpoints."""
    path = raw_path.split("?", 1)[0]
    m_upper = method.upper()

    def _json_resp(code: int, payload: Any) -> tuple[int, str, bytes]:
        return code, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8")

    if m_upper == "GET":
        if path in ("/", "/index.html"):
            return int(HTTPStatus.OK), "text/html; charset=utf-8", _WEB_CHAT_HTML.encode("utf-8")
        if path == "/health":
            return _json_resp(int(HTTPStatus.OK), {"status": "ok", "active_patches": registry.active_names()})
        if path == "/v1/models":
            now = int(time.time())
            ids = ["auto", base_model_name, *[p["name"] for p in registry.list_patches()]]
            seen: set[str] = set()
            rows = []
            for mid in ids:
                if mid in seen:
                    continue
                seen.add(mid)
                rows.append({"id": mid, "object": "model", "created": now, "owned_by": "wsi"})
            return _json_resp(int(HTTPStatus.OK), {"object": "list", "data": rows})
        if path == "/v1/wsi/patches":
            return _json_resp(
                int(HTTPStatus.OK),
                {"patches": registry.list_patches(), "active": registry.active_names()},
            )
        return _json_resp(int(HTTPStatus.NOT_FOUND), {"error": f"Unknown endpoint {path}"})

    if m_upper == "POST":
        try:
            body_obj = json.loads(raw_body.decode("utf-8")) if raw_body.strip() else {}
            if not isinstance(body_obj, dict):
                raise ValueError("JSON request body must be an object")

            if path == "/v1/wsi/patches/load":
                entry = registry.load_patch(
                    str(body_obj["name"]),
                    str(body_obj["path"]),
                    mode=str(body_obj.get("mode", "auto")),
                    activate=bool(body_obj.get("activate", False)),
                )
                return _json_resp(int(HTTPStatus.OK), {"patch": entry, "active": registry.active_names()})

            if path == "/v1/wsi/patches/activate":
                names = body_obj.get("patches", [])
                if not isinstance(names, list):
                    raise ValueError("'patches' must be a list of patch names")
                active = registry.activate(names, mode=body_obj.get("mode"))
                return _json_resp(int(HTTPStatus.OK), {"active": [a["name"] for a in active], "patches": active})

            if path == "/v1/wsi/patches/unload":
                removed = registry.unload_patch(str(body_obj["name"]))
                if not removed:
                    raise KeyError(f"Unknown patch {body_obj.get('name')!r}")
                return _json_resp(int(HTTPStatus.OK), {"removed": str(body_obj["name"]), "active": registry.active_names()})

            if path == "/v1/chat/completions":
                messages = body_obj.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("'messages' must be a non-empty list")
                if body_obj.get("stream"):
                    raise ValueError("Streaming responses are not yet supported; set stream=false")
                req_model = body_obj.get("model")
                req_patches = body_obj.get("patches")
                if req_patches is not None and not isinstance(req_patches, list):
                    raise ValueError("'patches' override must be a list of patch names")
                out = registry.generate_chat(
                    messages,
                    model_name=str(req_model) if req_model is not None else None,
                    patches=req_patches,
                    mode=str(body_obj["mode"]) if body_obj.get("mode") is not None else None,
                    max_tokens=int(body_obj.get("max_tokens", 256)),
                    temperature=float(body_obj.get("temperature", 0.0)),
                )
                resp = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": str(req_model or "auto"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": out["text"]},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": out["prompt_tokens"],
                        "completion_tokens": out["completion_tokens"],
                        "total_tokens": out["prompt_tokens"] + out["completion_tokens"],
                    },
                    "wsi": {
                        "active_patches": out["active_patches"],
                        "telemetry": out["wsi_telemetry"],
                    },
                }
                return _json_resp(int(HTTPStatus.OK), resp)

            return _json_resp(int(HTTPStatus.NOT_FOUND), {"error": f"Unknown endpoint {path}"})
        except (KeyError, ValueError, json.JSONDecodeError, FileNotFoundError) as exc:
            return _json_resp(int(HTTPStatus.BAD_REQUEST), {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return _json_resp(int(HTTPStatus.INTERNAL_SERVER_ERROR), {"error": str(exc)})

    return _json_resp(int(HTTPStatus.METHOD_NOT_ALLOWED), {"error": f"Method {m_upper} not allowed"})


def create_hotswap_server(
    registry: PatchRegistry,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    base_model_name: str = "base",
) -> ThreadingHTTPServer:
    """Create a thread-safe OpenAI-compatible + WSI hot-swap HTTP server with built-in web UI."""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _dispatch(self, method: str, raw_body: bytes = b"") -> None:
            code, ctype, payload = dispatch_hotswap_request(
                registry, method, self.path, raw_body, base_model_name=base_model_name
            )
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            if ctype.startswith("application/json"):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            n_bytes = int(self.headers.get("Content-Length", "0") or 0)
            raw_body = self.rfile.read(n_bytes) if n_bytes > 0 else b""
            self._dispatch("POST", raw_body)

    return ThreadingHTTPServer((host, int(port)), _Handler)


def serve_moipatch(
    model: Any,
    tokenizer: Any,
    *,
    patches: dict[str, MoIPatch | str | Path] | Sequence[MoIPatch | str | Path] | None = None,
    activate: Sequence[str] | None = None,
    mode: str = "auto",
    host: str = "127.0.0.1",
    port: int = 8000,
    base_model_name: str = "base",
    verify_fingerprint: bool = True,
) -> None:
    """Register patches, start the OpenAI-compatible + web chat hot-swap server, and block until Ctrl+C."""
    reg = PatchRegistry(model, tokenizer, verify_fingerprint=verify_fingerprint)
    loaded_names: list[str] = []
    if isinstance(patches, dict):
        for name, item in patches.items():
            reg.load_patch(name, item, mode=mode, activate=False)
            loaded_names.append(name)
    elif patches:
        for i, item in enumerate(patches):
            name = item.meta.get("name") or f"patch_{i}" if isinstance(item, MoIPatch) else Path(item).stem
            reg.load_patch(str(name), item, mode=mode, activate=False)
            loaded_names.append(str(name))

    to_activate = list(activate) if activate is not None else loaded_names
    if to_activate:
        reg.activate(to_activate, mode=mode)

    server = create_hotswap_server(reg, host=host, port=port, base_model_name=base_model_name)
    bound_host, bound_port = server.server_address[:2]
    print(f"[wsi serve] Web Chat UI : http://{bound_host}:{bound_port}/", flush=True)
    print(f"[wsi serve] OpenAI API  : http://{bound_host}:{bound_port}/v1", flush=True)
    print(f"[wsi serve] Loaded={loaded_names} | Active={reg.active_names()} | mode={mode}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
