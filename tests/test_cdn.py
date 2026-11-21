from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn

import wsi
from wsi.cdn import (
    INDEX_FILENAME,
    CDNCatalog,
    WeightCDNServer,
    WeightStreamClient,
    build_cdn_catalog,
    compute_model_down_proj_sha256,
)
from wsi.moimem import MemoryBank, _cached_sha256, _sha256
from wsi.moipatch import Edit, MoIPatch


class _LFM3MockLayer(nn.Module):
    def __init__(self, d_model: int = 8, d_ff: int = 16) -> None:
        super().__init__()

        class _FF(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w1 = nn.Linear(d_model, d_ff, bias=False)
                self.w3 = nn.Linear(d_model, d_ff, bias=False)
                self.w2 = nn.Linear(d_ff, d_model, bias=False)

        self.feed_forward = _FF()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.feed_forward.w2(h)


class _MockLFM3Model(nn.Module):
    def __init__(self, d_model: int = 8, d_ff: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_LFM3MockLayer(d_model, d_ff) for _ in range(2)]
        ).to(dtype=torch.bfloat16)


def test_cdn_exports() -> None:
    for sym in (
        "CDNCatalog",
        "CDNEntry",
        "StreamStats",
        "WeightCDNServer",
        "WeightStreamClient",
        "build_cdn_catalog",
        "compute_model_down_proj_sha256",
        "register_moipatch_entry",
        "shard_memory_bank",
    ):
        assert hasattr(wsi, sym), f"Missing export wsi.{sym}"


def test_cdn_streaming_memory_and_patch_e2e(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _MockLFM3Model(d_model=8, d_ff=16)
    sha_l0 = compute_model_down_proj_sha256(model, 0)
    sha_l1 = _sha256(model.layers[1].feed_forward.w2.weight.detach())

    # 4-person memory bank on layer 0 (`feed_forward.w2`)
    names = ["Alice Vorn", "Bob Merbernil", "Carol Kenwex", "Dave Quinyalu"]
    c = torch.zeros(4, 16, dtype=torch.float32)
    r = torch.zeros(4, 8, dtype=torch.float32)
    for idx in range(4):
        c[idx, idx] = 1.0
        r[idx, idx % 8] = float(idx + 1) * 0.5
    ck = torch.ones(4, dtype=torch.float32)
    bank = MemoryBank(
        layer=0,
        c=c,
        r=r,
        ck=ck,
        thr=0.5,
        names=names,
        meta={
            "method": "rome",
            "base_model": "mock-lfm3",
            "down_proj_sha256": sha_l0,
            "down_proj_dtype": "bfloat16",
        },
    )

    # Behavior `.moipatch` on layer 1
    u = torch.zeros(8, dtype=torch.float64)
    u[0] = 1.0
    patch = MoIPatch(
        edits=[
            Edit(
                module="layers.1.feed_forward.w2",
                u=u,
                alpha=1.0,
                idx=torch.tensor([0, 1], dtype=torch.long),
                base_sha256=sha_l1,
                base_dtype="bfloat16",
                shape=tuple(model.layers[1].feed_forward.w2.weight.shape),
            )
        ],
        meta={"name": "mock_prose_fix", "description": "coding surgical behavior fix", "base_model": "mock-lfm3"},
    )
    patch_file = patch.save(tmp_path / "src_patch.moipatch")

    cdn_dir = tmp_path / "cdn_store"
    cat = build_cdn_catalog(
        cdn_dir,
        memory_banks=[bank],
        moipatches=[(patch_file, ["surgical fix", "coding"])],
    )
    assert len(cat.entries) == 5

    # Reload catalog from disk & test local neighbor isolation
    cat_reloaded = CDNCatalog.load(cdn_dir)
    # "Mowex Merbernil" shares surname with "Bob Merbernil" -> must NOT match full-name memory shard
    assert cat_reloaded.resolve("Where does Mowex Merbernil live?", model_sha256=sha_l0) == []

    x_alice = torch.zeros(1, 1, 16, dtype=torch.bfloat16)
    x_alice[0, 0, 0] = 1.0
    x_orth = torch.zeros(1, 1, 16, dtype=torch.bfloat16)
    x_orth[0, 0, 9] = 1.0

    with torch.no_grad():
        base_alice_l0 = model.layers[0](x_alice).clone()
        base_orth_l0 = model.layers[0](x_orth).clone()
        base_l1 = model.layers[1](x_alice).clone()

    with WeightCDNServer(cat_reloaded, host="127.0.0.1", port=0) as srv:
        client = WeightStreamClient(srv.url, max_entries=8)

        # 1. Cold cache miss on Alice Vorn -> downloads ONLY Alice's ~1-row shard
        with client.stream_for("Where does Alice Vorn live?", model, model_sha256=sha_l0, kind="memory") as st:
            assert len(st.matched_ids) == 1
            assert st.cache_misses == 1
            assert st.cache_hits == 0
            assert 0 < st.bytes_downloaded < 8192
            with torch.no_grad():
                got_alice = model.layers[0](x_alice)
                got_orth = model.layers[0](x_orth)
                assert torch.equal(got_orth, base_orth_l0)
                assert (got_alice.float() - base_alice_l0.float() - r[0].view(1, 1, 8)).abs().max().item() < 0.05

        # Teardown guarantees base model restored
        with torch.no_grad():
            assert torch.equal(model.layers[0](x_alice), base_alice_l0)

        # 2. Warm cache hit on Alice Vorn -> 0 bytes transferred
        with client.stream_for("What job does Alice Vorn have?", model, model_sha256=sha_l0, kind="memory") as st2:
            assert len(st2.matched_ids) == 1
            assert st2.bytes_downloaded == 0
            assert st2.cache_hits == 1
            assert st2.cache_misses == 0
            with torch.no_grad():
                got_alice2 = model.layers[0](x_alice)
                assert (got_alice2.float() - base_alice_l0.float() - r[0].view(1, 1, 8)).abs().max().item() < 0.05

        # 3. Unrelated prompt -> 0 matches, 0 bytes
        with client.stream_for("What is 2 + 2?", model, model_sha256=sha_l0) as st3:
            assert st3.matched_ids == []
            assert st3.bytes_downloaded == 0
            with torch.no_grad():
                assert torch.equal(model.layers[0](x_alice), base_alice_l0)

        # 4. Wrong model SHA-256 -> 0 matches
        with client.stream_for("Where does Alice Vorn live?", model, model_sha256="deadbeef" * 8) as st4:
            assert st4.matched_ids == []
            assert st4.bytes_downloaded == 0

        # 5. Streaming a .moipatch behavioral patch + exact restoration on exit
        with client.stream_for("Apply surgical fix for coding task", model, model_sha256=sha_l1, kind="patch") as st5:
            assert st5.matched_ids == ["patch_mock_prose_fix"]
            assert st5.bytes_downloaded > 0
            with torch.no_grad():
                assert not torch.equal(model.layers[1](x_alice), base_l1)
        with torch.no_grad():
            assert torch.equal(model.layers[1](x_alice), base_l1)

        # 6. Tampered blob detection on cold fetch
        dave_entry = next(e for e in cat_reloaded.entries if "dave" in e.id)
        (cdn_dir / "blobs" / dave_entry.blob_relpath).write_bytes(b"corrupted-payload")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            with client.stream_for("Dave Quinyalu", model, model_sha256=sha_l0, kind="memory"):
                pass

    # 7. Simulated network latency check
    with WeightCDNServer(cat_reloaded, host="127.0.0.1", port=0, sim_latency_ms=30.0) as slow_srv:
        slow_client = WeightStreamClient(slow_srv.url)
        with slow_client.stream_for("Bob Merbernil", model, model_sha256=sha_l0, kind="memory") as st_slow:
            assert st_slow.bytes_downloaded > 0
            assert st_slow.download_ms >= 20.0


def test_cached_sha256_invalidates_on_mutation() -> None:
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16))
    d1 = _cached_sha256(p, torch.bfloat16)
    d2 = _cached_sha256(p, torch.bfloat16)
    assert d1 == d2 == _sha256(p.detach())

    with torch.no_grad():
        p[0, 0] += 1.0
    d3 = _cached_sha256(p, torch.bfloat16)
    assert d3 != d1
    assert d3 == _sha256(p.detach())

    p.data[0, 1] += 2.0
    d4 = _cached_sha256(p, torch.bfloat16)
    assert d4 != d3
    assert d4 == _sha256(p.detach())


def test_cdn_bf16_composition_live_edit_and_static_mode(tmp_path: Path) -> None:
    import functools
    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    torch.manual_seed(1)
    model = _MockLFM3Model(d_model=8, d_ff=256)
    sha_l0 = compute_model_down_proj_sha256(model, 0)
    sha_l1 = _sha256(model.layers[1].feed_forward.w2.weight.detach())

    names = ["Alice Vorn", "Bob Merbernil"]
    c = torch.zeros(2, 256, dtype=torch.float32)
    r = torch.zeros(2, 8, dtype=torch.float32)
    c[0, 0] = 1.0
    r[0, 0] = 2.5
    c[1, 1] = 1.0
    r[1, 1] = 3.5
    bank = MemoryBank(
        layer=0,
        c=c,
        r=r,
        ck=torch.ones(2, dtype=torch.float32),
        thr=0.5,
        names=names,
        meta={"method": "rome", "base_model": "mock-lfm3", "down_proj_sha256": sha_l0, "down_proj_dtype": "bfloat16"},
    )

    # Verify bf16 shards are substantially smaller than fp32 shards
    fp32_dir = tmp_path / "fp32_store"
    bf16_dir = tmp_path / "bf16_store"
    cat_fp32 = build_cdn_catalog(fp32_dir, memory_banks=[bank], shard_dtype="fp32")
    assert (bf16_dir / INDEX_FILENAME).exists() is False

    u = torch.zeros(8, dtype=torch.float64)
    u[0] = 1.0
    patch = MoIPatch(
        edits=[
            Edit(
                module="layers.1.feed_forward.w2",
                u=u,
                alpha=1.0,
                idx=torch.tensor([0, 1], dtype=torch.long),
                base_sha256=sha_l1,
                base_dtype="bfloat16",
                shape=tuple(model.layers[1].feed_forward.w2.weight.shape),
            )
        ],
        meta={"name": "mock_prose_fix", "description": "coding surgical behavior fix", "base_model": "mock-lfm3"},
    )
    patch_file = patch.save(tmp_path / "p.moipatch")
    cat_bf16 = build_cdn_catalog(
        bf16_dir,
        memory_banks=[bank],
        moipatches=[(patch_file, ["surgical fix", "coding"])],
        shard_dtype="bf16",
    )
    assert (bf16_dir / INDEX_FILENAME).is_file()
    fp32_size = cat_fp32.entries[0].size_bytes
    bf16_size = next(e for e in cat_bf16.entries if e.kind == "memory").size_bytes
    assert bf16_size < fp32_size

    x_alice = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
    x_alice[0, 0, 0] = 1.0
    x_bob = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
    x_bob[0, 0, 1] = 1.0
    x_orth = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
    x_orth[0, 0, 9] = 1.0
    with torch.no_grad():
        base_alice_l0 = model.layers[0](x_alice).clone()
        base_bob_l0 = model.layers[0](x_bob).clone()
        base_orth_l0 = model.layers[0](x_orth).clone()
        base_l1 = model.layers[1](x_alice).clone()

    with WeightCDNServer(cat_bf16, host="127.0.0.1", port=0) as srv:
        client = WeightStreamClient(srv.url, max_entries=16)

        # 1. Live composition: 2 memories (L0) + 1 .moipatch (L1) in one prompt
        with client.stream_for(
            "Compare Alice Vorn and Bob Merbernil with surgical fix",
            model,
            model_sha256={sha_l0, sha_l1},
            top_k=4,
        ) as st_comp:
            assert len(st_comp.matched_ids) == 3
            assert sorted(st_comp.kinds) == ["memory", "memory", "patch"]
            with torch.no_grad():
                assert (model.layers[0](x_alice).float() - base_alice_l0.float() - r[0].view(1, 1, 8)).abs().max().item() < 0.05
                assert (model.layers[0](x_bob).float() - base_bob_l0.float() - r[1].view(1, 1, 8)).abs().max().item() < 0.05
                assert torch.equal(model.layers[0](x_orth), base_orth_l0)
                assert not torch.equal(model.layers[1](x_alice), base_l1)
        with torch.no_grad():
            assert torch.equal(model.layers[0](x_alice), base_alice_l0)
            assert torch.equal(model.layers[1](x_alice), base_l1)

        # 2. Live Knowledge Edit without restarting server: mutate Alice Vorn's value vector in place on disk
        r_new = torch.zeros(1, 8, dtype=torch.float32)
        r_new[0, 0] = -4.0
        new_alice = MemoryBank(
            layer=0,
            c=c[0:1],
            r=r_new,
            ck=torch.ones(1, dtype=torch.float32),
            thr=0.5,
            names=["Alice Vorn"],
            meta={"method": "rome", "base_model": "mock-lfm3", "down_proj_sha256": sha_l0, "down_proj_dtype": "bfloat16"},
        )
        cat_bf16.upsert_memory_shard(new_alice, name="Alice Vorn", dtype="bf16")

        # Next query must auto-invalidate Alice (SHA changed -> miss + download), while Bob stays cached
        with client.stream_for(
            "Compare Alice Vorn and Bob Merbernil",
            model,
            model_sha256=sha_l0,
            kind="memory",
            top_k=4,
        ) as st_edit:
            assert st_edit.cache_misses == 1
            assert st_edit.cache_hits == 1
            assert st_edit.bytes_downloaded == bf16_size or st_edit.bytes_downloaded > 0
            with torch.no_grad():
                assert (model.layers[0](x_alice).float() - base_alice_l0.float() - r_new.view(1, 1, 8)).abs().max().item() < 0.05
                assert (model.layers[0](x_bob).float() - base_bob_l0.float() - r[1].view(1, 1, 8)).abs().max().item() < 0.05

    # 3. Static CDN mode (plain file server like GitHub Raw / Cloudflare R2, zero API server)
    class _QuietStaticHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

    handler_cls = functools.partial(_QuietStaticHandler, directory=str(bf16_dir))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        port = httpd.server_address[1]
        static_client = WeightStreamClient(f"http://127.0.0.1:{port}", mode="static")
        with static_client.stream_for("Where does Alice Vorn live?", model, model_sha256=sha_l0, kind="memory") as st_s1:
            assert len(st_s1.matched_ids) == 1
            assert st_s1.cache_misses == 1
            assert st_s1.bytes_downloaded > 0
            with torch.no_grad():
                assert (model.layers[0](x_alice).float() - base_alice_l0.float() - r_new.view(1, 1, 8)).abs().max().item() < 0.05

        with static_client.stream_for("Alice Vorn", model, model_sha256=sha_l0, kind="memory") as st_s2:
            assert st_s2.cache_hits == 1
            assert st_s2.bytes_downloaded == 0
    finally:
        httpd.shutdown()
        httpd.server_close()
        th.join(timeout=2.0)
