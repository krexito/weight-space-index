"""CPU unit tests for checkpoint baking (`bake_moipatch`) and live hot-swap server (`PatchRegistry` / `create_hotswap_server`)."""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
from wsi.moipatch import Edit, MoIPatch, _main as moipatch_main, _sha256, apply_edit  # noqa: E402
from wsi.serve import PatchRegistry, bake_moipatch, create_hotswap_server, dispatch_hotswap_request  # noqa: E402


def _make_toy_patches(hidden: int) -> tuple[MoIPatch, MoIPatch, Edit]:
    u_ref = torch.zeros(hidden, dtype=torch.float64)
    u_ref[0] = 1.0
    r_dir = torch.zeros(hidden, dtype=torch.float32)
    r_dir[3] = 1.0
    v_dir = torch.zeros(hidden, dtype=torch.float32)
    v_dir[4] = 1.0
    edit = Edit(module="layers.2.self_attn.o_proj", u=u_ref, alpha=1.5)
    p_ablit = MoIPatch(
        edits=[edit],
        router={"layer": 3, "readout": "last_token", "dir": r_dir, "thr": 1.0},
        meta={"name": "abliterated"},
    )
    p_sfw = MoIPatch(
        edits=[edit],
        router={"layer": 3, "readout": "last_token", "dir": r_dir, "thr": 1.0, "veto_dir": v_dir, "veto_thr": 1.0},
        meta={"name": "abliterated-sfw"},
    )
    return p_ablit, p_sfw, edit


def test_bake_moipatch_single_and_sharded(tmp_path: Path) -> None:
    torch.manual_seed(0)
    w0 = torch.randn(16, 16, dtype=torch.float32)
    w1 = torch.randn(16, 16, dtype=torch.float32)
    w_other = torch.randn(16, 16, dtype=torch.float32)

    # 1. Single-file checkpoint + sequential bake of 2 static patches
    single_src = tmp_path / "single_src"
    single_src.mkdir()
    save_file(
        {
            "model.layers.0.self_attn.o_proj.weight": w0,
            "model.layers.1.mlp.down_proj.weight": w_other,
        },
        str(single_src / "model.safetensors"),
        metadata={"format": "pt"},
    )
    (single_src / "config.json").write_text('{"hidden_size": 16}', encoding="utf-8")

    u0 = torch.zeros(16, dtype=torch.float64)
    u0[2] = 1.0
    idx0 = torch.tensor([0, 3, 7], dtype=torch.long)
    e0 = Edit(module="layers.0.self_attn.o_proj", u=u0, alpha=0.8, idx=idx0, base_sha256=_sha256(w0), base_dtype="float32", shape=(16, 16))
    p_static_a = MoIPatch(edits=[e0], meta={"name": "static_a"})

    u1 = torch.zeros(16, dtype=torch.float64)
    u1[5] = 1.0
    e1 = Edit(module="layers.0.self_attn.o_proj", u=u1, alpha=1.2, base_dtype="float32", shape=(16, 16))
    p_static_b = MoIPatch(edits=[e1], meta={"name": "static_b"})

    single_out = tmp_path / "single_out"
    rep_s = bake_moipatch(single_src, [p_static_a, p_static_b], single_out, copy_untouched="copy")
    assert rep_s["n_touched_shards"] == 1 and rep_s["n_total_shards"] == 1
    assert rep_s["had_router_warning"] is False
    assert (single_out / "config.json").read_text(encoding="utf-8") == '{"hidden_size": 16}'
    assert (single_out / "wsi_baked_patches.json").exists()

    expected_w0 = apply_edit(apply_edit(w0, u0, idx0, 0.8), u1, None, 1.2)
    with safe_open(str(single_out / "model.safetensors"), "pt") as sf:
        baked_w0 = sf.get_tensor("model.layers.0.self_attn.o_proj.weight").clone()
        untouched_w = sf.get_tensor("model.layers.1.mlp.down_proj.weight").clone()
    assert torch.equal(baked_w0, expected_w0)
    assert torch.equal(untouched_w, w_other)

    # 2. Sharded checkpoint (only shard 1 touched; shard 2 untouched)
    shard_src = tmp_path / "shard_src"
    shard_src.mkdir()
    save_file({"model.layers.0.self_attn.o_proj.weight": w0}, str(shard_src / "model-00001-of-00002.safetensors"))
    save_file({"model.layers.1.mlp.down_proj.weight": w1}, str(shard_src / "model-00002-of-00002.safetensors"))
    idx_json = {
        "weight_map": {
            "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
        }
    }
    (shard_src / "model.safetensors.index.json").write_text(json.dumps(idx_json), encoding="utf-8")

    shard_out = tmp_path / "shard_out"
    rep_sh = bake_moipatch(shard_src, p_static_a, shard_out, copy_untouched="copy")
    assert rep_sh["n_touched_shards"] == 1 and rep_sh["n_total_shards"] == 2
    with safe_open(str(shard_out / "model-00001-of-00002.safetensors"), "pt") as sf:
        assert torch.equal(sf.get_tensor("model.layers.0.self_attn.o_proj.weight").clone(), apply_edit(w0, u0, idx0, 0.8))
    with safe_open(str(shard_out / "model-00002-of-00002.safetensors"), "pt") as sf:
        assert torch.equal(sf.get_tensor("model.layers.1.mlp.down_proj.weight").clone(), w1)

    # 3. Fingerprint mismatch must raise
    bad_edit = Edit(module="layers.0.self_attn.o_proj", u=u0, alpha=0.8, base_sha256="deadbeef" * 8, base_dtype="float32")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bake_moipatch(single_src, MoIPatch(edits=[bad_edit]), tmp_path / "bad_out")

    # 4. Routed patch guard unless allow_routed=True
    r_dir = torch.zeros(16, dtype=torch.float32)
    r_dir[0] = 1.0
    p_routed = MoIPatch(edits=[e0], router={"layer": 1, "readout": "last_token", "dir": r_dir, "thr": 0.5}, meta={"name": "abliterated-sfw"})
    with pytest.raises(ValueError, match="runtime router"):
        bake_moipatch(single_src, p_routed, tmp_path / "routed_fail")

    rep_forced = bake_moipatch(single_src, p_routed, tmp_path / "routed_ok", allow_routed=True, copy_untouched="copy")
    assert rep_forced["had_router_warning"] is True
    assert rep_forced["routed_patches"] == ["abliterated-sfw"]


def test_patch_registry_hotswap_and_per_request_override() -> None:
    torch.manual_seed(0)
    model = sr._TinyLFM3ToyLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = sr._TinyRefusalTokenizer()
    p_ablit, p_sfw, _ = _make_toy_patches(model.config.hidden_size)

    reg = PatchRegistry(model, tok)
    reg.load_patch("abliterated", p_ablit, mode="auto", activate=False)
    reg.load_patch("abliterated-sfw", p_sfw, mode="auto", activate=False)

    # 1. Base mode: both trigger & toxic refuse
    out_base_trig = reg.generate_chat([{"role": "user", "content": "trigger:0 benign"}], max_tokens=3)
    out_base_tox = reg.generate_chat([{"role": "user", "content": "toxic:0 harmful"}], max_tokens=3)
    assert out_base_trig["active_patches"] == []
    assert sr._is_refusal(out_base_trig["text"])
    assert sr._is_refusal(out_base_tox["text"])

    # 2. Hot-swap to `abliterated`: unblocks BOTH trigger and toxic immediately
    reg.activate(["abliterated"])
    out_ab_trig = reg.generate_chat([{"role": "user", "content": "trigger:0 benign"}], max_tokens=3)
    out_ab_tox = reg.generate_chat([{"role": "user", "content": "toxic:0 harmful"}], max_tokens=3)
    assert out_ab_trig["active_patches"] == ["abliterated"]
    assert not sr._is_refusal(out_ab_trig["text"])
    assert not sr._is_refusal(out_ab_tox["text"])

    # 3. Hot-swap to `abliterated-sfw`: unblocks trigger, preserves toxic refusal, keeps control intact
    reg.activate(["abliterated-sfw"])
    out_sfw_trig = reg.generate_chat([{"role": "user", "content": "trigger:0 benign"}], max_tokens=3)
    out_sfw_tox = reg.generate_chat([{"role": "user", "content": "toxic:0 harmful"}], max_tokens=3)
    out_sfw_ctrl = reg.generate_chat([{"role": "user", "content": "normal:0 hello"}], max_tokens=3)
    assert not sr._is_refusal(out_sfw_trig["text"])
    assert sr._is_refusal(out_sfw_tox["text"])
    assert out_sfw_ctrl["wsi_telemetry"]["abliterated-sfw"]["fired"] == 0

    # 4. Per-request overrides restore active set cleanly
    ov_base = reg.generate_chat([{"role": "user", "content": "trigger:0 benign"}], model_name="base", max_tokens=3)
    ov_ablit = reg.generate_chat([{"role": "user", "content": "toxic:0 harmful"}], model_name="abliterated", max_tokens=3)
    assert sr._is_refusal(ov_base["text"]) and ov_base["active_patches"] == []
    assert not sr._is_refusal(ov_ablit["text"]) and ov_ablit["active_patches"] == ["abliterated"]
    assert reg.active_names() == ["abliterated-sfw"]


def _http_req(url: str, payload: dict | None = None) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8")


def test_hotswap_http_server_openai_and_ui(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = sr._TinyLFM3ToyLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = sr._TinyRefusalTokenizer()
    p_ablit, p_sfw, _ = _make_toy_patches(model.config.hidden_size)
    ablit_path = tmp_path / "ablit.moipatch"
    p_ablit.save(ablit_path)

    reg = PatchRegistry(model, tok)
    reg.load_patch("abliterated-sfw", p_sfw, mode="auto", activate=False)

    def _call(method: str, path: str, payload: dict | None = None) -> tuple[int, str]:
        raw = json.dumps(payload).encode("utf-8") if payload is not None else b""
        code, _ctype, out_b = dispatch_hotswap_request(reg, method, path, raw, base_model_name="tiny-lfm3")
        return code, out_b.decode("utf-8")

    # 1. Exercise full HTTP/OpenAI/UI router deterministically
    code, html = _call("GET", "/")
    assert code == 200 and "WSI Hot-Swap Studio" in html

    code, health = _call("GET", "/health")
    assert code == 200 and json.loads(health)["active_patches"] == []

    code, models_s = _call("GET", "/v1/models")
    model_ids = {m["id"] for m in json.loads(models_s)["data"]}
    assert {"auto", "tiny-lfm3", "abliterated-sfw"}.issubset(model_ids)

    code, plist = _call("GET", "/v1/wsi/patches")
    assert code == 200 and len(json.loads(plist)["patches"]) == 1

    # Base active: trigger refuses
    code, c1 = _call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "trigger:0 x"}], "max_tokens": 3})
    assert code == 200
    assert sr._is_refusal(json.loads(c1)["choices"][0]["message"]["content"])

    # Live hot-swap -> activate `abliterated-sfw`
    code, act = _call("POST", "/v1/wsi/patches/activate", {"patches": ["abliterated-sfw"]})
    assert code == 200 and json.loads(act)["active"] == ["abliterated-sfw"]

    code, c2 = _call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "trigger:0 x"}], "max_tokens": 3})
    d2 = json.loads(c2)
    assert code == 200
    assert not sr._is_refusal(d2["choices"][0]["message"]["content"])
    assert d2["wsi"]["active_patches"] == ["abliterated-sfw"]

    # Load & activate saved patch dynamically, then unload
    code, ld = _call("POST", "/v1/wsi/patches/load", {"name": "abliterated", "path": str(ablit_path), "activate": True})
    assert code == 200 and "abliterated" in json.loads(ld)["active"]

    code, unl = _call("POST", "/v1/wsi/patches/unload", {"name": "abliterated"})
    assert code == 200 and "abliterated" not in json.loads(unl)["active"]

    # Error handling checks
    assert _call("GET", "/no-such-route")[0] == 404
    assert _call("POST", "/v1/chat/completions", {"messages": []})[0] == 400

    # 2. Verify ThreadingHTTPServer instantiation & port binding cleanly
    server = create_hotswap_server(reg, host="127.0.0.1", port=0, base_model_name="tiny-lfm3")
    port = server.server_address[1]
    assert port > 0
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        try:
            code_live, body_live = _http_req(f"http://127.0.0.1:{port}/health")
            assert code_live == 200 and json.loads(body_live)["status"] == "ok"
        except (urllib.error.URLError, PermissionError):
            pass  # Loopback socket connect may be blocked in restricted Windows sandbox
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=2)


def test_moipatch_cli_export_subcommand(tmp_path: Path, capsys) -> None:
    torch.manual_seed(1)
    w = torch.randn(12, 12, dtype=torch.float32)
    src = tmp_path / "cli_src"
    src.mkdir()
    save_file({"model.layers.0.mlp.down_proj.weight": w}, str(src / "model.safetensors"))

    u = torch.zeros(12, dtype=torch.float64)
    u[1] = 1.0
    p_path = tmp_path / "cli_static.moipatch"
    MoIPatch(edits=[Edit(module="layers.0.mlp.down_proj", u=u, alpha=0.5, base_sha256=_sha256(w), base_dtype="float32")]).save(p_path)

    out_dir = tmp_path / "cli_out"
    moipatch_main(["export", str(p_path), "--model", str(src), "--out", str(out_dir), "--copy", "--json"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["n_touched_shards"] == 1
    with safe_open(str(out_dir / "model.safetensors"), "pt") as sf:
        assert torch.equal(sf.get_tensor("model.layers.0.mlp.down_proj.weight").clone(), apply_edit(w, u, None, 0.5))
