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
    CDNCatalog,
    WeightCDNServer,
    WeightStreamClient,
    build_cdn_catalog,
    compute_model_down_proj_sha256,
)
from wsi.moimem import MemoryBank, _sha256
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
