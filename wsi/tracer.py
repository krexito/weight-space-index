"""Token-level dense & compact activation tracing (`trace_all` forensics + block-quantized `CompactTrace`).

Pairs `wsi.capture.trace_all` with `WeightSpaceIndex` geometry when global averages fail:
  1. `diff_traces(model, ids_a, ids_b, wsi=...)`: full-fidelity X-ray via `trace_all` across two prompts
     (e.g., neutral vs. pushback turn, clean vs. trigger prompt). Handles unequal prompt lengths and
     ranks exact layers, attention heads, MLP neurons (`Δact_i * ||W_down[:, i]||`), cache state shifts,
     attention redistribution, and vocabulary readout (`explain_vector`).
  2. `capture_compact(model, ids, ...)`: scalable token-by-token tracing using micro-block INT4/FP8
     quantization on residual streams + Top-K sparse contribution-weighted MLP neurons + per-head L2
     energy, compressing full sequences ~10-40x vs fp32 `trace_all` while preserving loop/drift onset
     forensics (`detect_loop_onset`, `compare_positions`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .capture import _is_linear, find_blocks, find_mlps, trace_all


# ----------------------------------------------------------------------------------------------- block quantization
def quantize_block_int4(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Micro-block symmetric INT4 quantization (two signed nibbles [-7..7] packed per uint8 byte + fp16 scale).

    Per-group scaling (`group_size=16`, matching micro-scaled FP4 conventions) isolates large outlier
    channels to their own 16-element group instead of crushing an entire hidden vector to zero.
    Returns (packed_uint8, scales_fp16, orig_shape).
    """
    if group_size % 2 != 0:
        raise ValueError("group_size must be even for 2-nibble uint8 packing")
    orig_shape = tuple(x.shape)
    flat = x.detach().float().reshape(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=torch.float32)])
    blocks = flat.reshape(-1, group_size)
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    scales = (max_abs / 7.0).clamp_min(1e-8)
    q = torch.round(blocks / scales).clamp(-7, 7).to(torch.int8) + 8  # map [-7..7] -> [1..15]
    q_u8 = q.to(torch.uint8).reshape(-1, 2)
    packed = ((q_u8[:, 0] & 0x0F) | ((q_u8[:, 1] & 0x0F) << 4)).cpu()
    return packed, scales.squeeze(1).to(torch.float16).cpu(), orig_shape


def dequantize_block_int4(packed: torch.Tensor, scales: torch.Tensor, orig_shape: tuple[int, ...],
                          group_size: int = 16) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int8) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    q = torch.stack([lo, hi], dim=1).reshape(-1, group_size).float()
    deq = (q * scales.float().unsqueeze(1)).reshape(-1)
    n = 1
    for s in orig_shape:
        n *= s
    return deq[:n].reshape(orig_shape)


def quantize_fp8_or_int8(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    """Quantize to 1 byte/element (micro-block INT8 for portability; scale isolates outliers)."""
    flat = x.detach().float().reshape(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=torch.float32)])
    blocks = flat.reshape(-1, group_size)
    scales = (blocks.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
    q = torch.round(blocks / scales).clamp(-127, 127).to(torch.int8).cpu()
    return q.reshape(-1), scales.squeeze(1).to(torch.float16).cpu(), "int8_blk"


def dequantize_fp8_or_int8(q: torch.Tensor, scales: torch.Tensor, orig_shape: tuple[int, ...],
                           group_size: int = 16) -> torch.Tensor:
    deq = (q.reshape(-1, group_size).float() * scales.float().unsqueeze(1)).reshape(-1)
    n = 1
    for s in orig_shape:
        n *= s
    return deq[:n].reshape(orig_shape)


def quantization_error(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, float]:
    xf, hf = x.detach().float().cpu().reshape(-1), x_hat.detach().float().cpu().reshape(-1)
    diff = (xf - hf).norm().item()
    norm = max(xf.norm().item(), 1e-12)
    cos = float(torch.nn.functional.cosine_similarity(xf.unsqueeze(0), hf.unsqueeze(0)).item()) if norm > 1e-12 else 1.0
    return {"rel_l2": round(diff / norm, 6), "cosine": round(cos, 6)}


# ----------------------------------------------------------------------------------------------- helpers
def _layer_idx_from_name(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    m = re.search(r"\.h\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _infer_attn_heads(model: nn.Module, out_proj_name: str, d_in: int) -> int:
    mods = dict(model.named_modules())
    parent_name = out_proj_name.rsplit(".", 1)[0] if "." in out_proj_name else ""
    parent = mods.get(parent_name)
    if parent is not None:
        for attr in ("num_v_heads", "linear_num_value_heads", "num_heads", "num_attention_heads", "n_head"):
            v = getattr(parent, attr, None)
            if isinstance(v, int) and v > 0 and d_in % v == 0:
                return v
        for hd_attr in ("head_v_dim", "head_dim", "linear_value_head_dim"):
            hd = getattr(parent, hd_attr, None)
            if isinstance(hd, int) and hd > 0 and d_in % hd == 0:
                return d_in // hd
    cfg = getattr(getattr(model, "config", None), "text_config", getattr(model, "config", None))
    if cfg is not None:
        if "linear_attn" in out_proj_name:
            v = getattr(cfg, "linear_num_value_heads", None)
            if isinstance(v, int) and v > 0 and d_in % v == 0:
                return v
        for attr in ("num_attention_heads", "n_head"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0 and d_in % v == 0:
                return v
    return 1


def _find_attn_out_projs(model: nn.Module, hidden: int) -> dict[str, tuple[nn.Module, int]]:
    out = {}
    for n, m in model.named_modules():
        if "mtp" in n or "visual" in n or not _is_linear(m):
            continue
        if n.endswith((".o_proj", ".out_proj", ".c_proj")) and ".mlp." not in n:
            nh = _infer_attn_heads(model, n, m.weight.shape[1] if isinstance(m, nn.Linear) else m.weight.shape[0])
            out[n] = (m, nh)
    return out


# ----------------------------------------------------------------------------------------------- CompactTrace
@dataclass
class CompactTrace:
    """Scalable token-by-token activation trace combining micro-block INT4/INT8 residuals, Top-K MLP units,
    and per-head attention energies across full sequences."""

    seq_len: int
    n_layers: int
    hidden_size: int
    resid_quant: str
    group_size: int
    top_k_mlp: int
    resid: dict[int, dict[str, Any]]  # layer -> {"data": Tensor, "scales": Tensor | None, "shape": tuple}
    mlp_topk: dict[int, dict[str, torch.Tensor]]  # layer -> {"idx": int32[T, K], "val": fp16[T, K], "energy_frac": fp16[T]}
    head_norms: dict[str, dict[str, Any]]  # mod_name -> {"layer": int|None, "n_heads": int, "l2": fp16[T, H]}
    meta: dict[str, Any] = field(default_factory=dict)

    def dequant_resid(self, layer: int) -> torch.Tensor:
        r = self.resid[layer]
        if self.resid_quant == "int4":
            return dequantize_block_int4(r["data"], r["scales"], tuple(r["shape"]), self.group_size)
        if self.resid_quant == "int8":
            return dequantize_fp8_or_int8(r["data"], r["scales"], tuple(r["shape"]), self.group_size)
        return r["data"].float()

    def size_bytes(self) -> int:
        b = sum(r["data"].numel() * r["data"].element_size()
                + (r["scales"].numel() * r["scales"].element_size() if r["scales"] is not None else 0)
                for r in self.resid.values())
        b += sum(t.numel() * t.element_size() for m in self.mlp_topk.values() for t in m.values())
        b += sum(h["l2"].numel() * h["l2"].element_size() for h in self.head_norms.values())
        return b

    def compare_positions(self, t_target: int, t_ref: int, *, top_k: int = 8) -> dict[str, Any]:
        """Find which layers, MLP neurons, and attention heads shifted strongest at token `t_target` vs `t_ref`."""
        layer_shifts, mlp_shifts, head_shifts = [], [], []
        for l in sorted(self.resid):
            R = self.dequant_resid(l)
            va, vb = R[t_ref], R[t_target]
            layer_shifts.append({
                "layer": l,
                "delta_norm": round(float((vb - va).norm()), 4),
                "cosine": round(float(torch.nn.functional.cosine_similarity(va[None], vb[None])), 5),
            })
        for l, m in sorted(self.mlp_topk.items()):
            ia, va = m["idx"][t_ref].tolist(), m["val"][t_ref].float().tolist()
            ib, vb = m["idx"][t_target].tolist(), m["val"][t_target].float().tolist()
            ma, mb = dict(zip(ia, va)), dict(zip(ib, vb))
            for u in set(ma) | set(mb):
                da, db = ma.get(u, 0.0), mb.get(u, 0.0)
                mlp_shifts.append({"layer": l, "unit": int(u), "ref_contrib": round(da, 4),
                                   "target_contrib": round(db, 4), "delta": round(db - da, 4),
                                   "abs_delta": round(abs(db - da), 4)})
        for mod, h in self.head_norms.items():
            da = h["l2"][t_ref].float()
            db = h["l2"][t_target].float()
            diff = db - da
            for hid in diff.abs().topk(min(3, len(diff))).indices.tolist():
                head_shifts.append({"module": mod, "layer": h["layer"], "head": hid,
                                    "ref_l2": round(float(da[hid]), 4), "target_l2": round(float(db[hid]), 4),
                                    "delta": round(float(diff[hid]), 4), "abs_delta": round(float(diff[hid].abs()), 4)})
        mlp_shifts.sort(key=lambda x: -x["abs_delta"])
        head_shifts.sort(key=lambda x: -x["abs_delta"])
        return {
            "t_target": t_target, "t_ref": t_ref,
            "layer_shifts": sorted(layer_shifts, key=lambda x: -x["delta_norm"])[:top_k],
            "top_mlp_shifts": mlp_shifts[:top_k],
            "top_head_shifts": head_shifts[:top_k],
        }

    def detect_loop_onset(self, layer: int | None = None, window: int = 8, thr: float = 0.96) -> dict[str, Any]:
        """Scan token-to-previous-token residual cosine inside one trace to locate repetition onset `t*`."""
        l = (max(self.resid) // 2) if layer is None else layer
        R = self.dequant_resid(l)
        Rn = R / R.norm(dim=1, keepdim=True).clamp_min(1e-8)
        T = Rn.shape[0]
        if T <= window:
            return {"layer": l, "onset_token": None, "max_lag_cos": 0.0}
        lag_cos = torch.zeros(T)
        for t in range(1, T):
            prev = Rn[max(0, t - window) : t]
            lag_cos[t] = (prev @ Rn[t]).max()
        hits = (lag_cos > thr).nonzero(as_tuple=True)[0].tolist()
        return {
            "layer": l,
            "onset_token": int(hits[0]) if hits else None,
            "max_lag_cos": round(float(lag_cos.max()), 5),
            "mean_tail_cos": round(float(lag_cos[T // 2:].mean()), 5) if T >= 4 else round(float(lag_cos.mean()), 5),
        }

    def save(self, out_dir: str | Path) -> Path:
        from safetensors.torch import save_file

        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        flat: dict[str, torch.Tensor] = {}
        resid_shapes = {}
        for l, r in self.resid.items():
            flat[f"resid.{l}.data"] = r["data"].contiguous()
            if r["scales"] is not None:
                flat[f"resid.{l}.scales"] = r["scales"].contiguous()
            resid_shapes[str(l)] = list(r["shape"])
        for l, m in self.mlp_topk.items():
            for k, v in m.items():
                flat[f"mlp.{l}.{k}"] = v.contiguous()
        head_meta = {}
        for idx, (mod, h) in enumerate(self.head_norms.items()):
            flat[f"head.{idx}"] = h["l2"].contiguous()
            head_meta[str(idx)] = {"module": mod, "layer": h["layer"], "n_heads": h["n_heads"]}
        save_file(flat, str(p / "compact_trace.safetensors"))
        (p / "compact_trace.json").write_text(json.dumps({
            "seq_len": self.seq_len, "n_layers": self.n_layers, "hidden_size": self.hidden_size,
            "resid_quant": self.resid_quant, "group_size": self.group_size, "top_k_mlp": self.top_k_mlp,
            "resid_shapes": resid_shapes, "heads": head_meta, "meta": self.meta,
        }, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, trace_dir: str | Path) -> "CompactTrace":
        from safetensors.torch import load_file

        p = Path(trace_dir)
        info = json.loads((p / "compact_trace.json").read_text(encoding="utf-8"))
        tens = load_file(str(p / "compact_trace.safetensors"))
        resid, mlp_topk, head_norms = {}, {}, {}
        for l_str, shp in info["resid_shapes"].items():
            l = int(l_str)
            resid[l] = {"data": tens[f"resid.{l}.data"], "scales": tens.get(f"resid.{l}.scales"), "shape": shp}
        for l in range(info["n_layers"]):
            if f"mlp.{l}.idx" in tens:
                mlp_topk[l] = {"idx": tens[f"mlp.{l}.idx"], "val": tens[f"mlp.{l}.val"],
                               "energy_frac": tens[f"mlp.{l}.energy_frac"]}
        for idx, hm in info["heads"].items():
            head_norms[hm["module"]] = {"layer": hm["layer"], "n_heads": hm["n_heads"], "l2": tens[f"head.{idx}"]}
        return cls(info["seq_len"], info["n_layers"], info["hidden_size"], info["resid_quant"],
                   info["group_size"], info["top_k_mlp"], resid, mlp_topk, head_norms, info.get("meta", {}))


@torch.no_grad()
def capture_compact(
    model: nn.Module,
    input_ids: torch.Tensor | list[int],
    *,
    top_k_mlp: int = 256,
    resid_quant: str = "int8",
    group_size: int = 16,
    max_tokens: int | None = None,
) -> CompactTrace:
    """Record a compressed token-by-token trace: quantized residuals, Top-K weighted MLP units, and head L2 norms."""
    dev = next(model.parameters()).device
    ids = (input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)).reshape(1, -1)
    if max_tokens is not None:
        ids = ids[:, :max_tokens]
    T = ids.shape[1]

    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    mods = dict(model.named_modules())
    mlps = find_mlps(model)
    down_norms = {}
    mlp_by_down = {}
    for g in mlps:
        m = mods[g.down]
        w = m.weight.detach().float()
        down_norms[g.down] = w.norm(dim=0) if isinstance(m, nn.Linear) else w.norm(dim=1)
        li = _layer_idx_from_name(g.down)
        mlp_by_down[g.down] = li if li is not None else len(mlp_by_down)

    # Infer hidden width cleanly from the first available linear out_features
    d0 = mods[mlps[0].down]
    hidden = d0.weight.shape[0] if isinstance(d0, nn.Linear) else d0.weight.shape[1]
    attns = _find_attn_out_projs(model, hidden)

    resid_out: dict[int, dict[str, Any]] = {}
    mlp_topk_out: dict[int, dict[str, torch.Tensor]] = {}
    head_out: dict[str, dict[str, Any]] = {}
    handles = []

    for l, blk in enumerate(blocks):
        def bhook(mod, inp, out, l=l):
            t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor):
                return
            x = t.detach().reshape(-1, t.shape[-1])  # [T, H]
            if resid_quant == "int4":
                packed, sc, shp = quantize_block_int4(x, group_size)
                resid_out[l] = {"data": packed, "scales": sc, "shape": list(shp)}
            elif resid_quant == "int8":
                q, sc, _ = quantize_fp8_or_int8(x, group_size)
                resid_out[l] = {"data": q, "scales": sc, "shape": list(x.shape)}
            else:
                resid_out[l] = {"data": x.to(torch.float16).cpu(), "scales": None, "shape": list(x.shape)}

        handles.append(blk.register_forward_hook(bhook))

    for dname, li in mlp_by_down.items():
        m = mods[dname]
        wn = down_norms[dname]

        def mhook(mod, inp, out, li=li, wn=wn):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()  # [T, U]
            contrib = x * wn.to(x.device)
            k = min(top_k_mlp, contrib.shape[1])
            vals, idx = contrib.abs().topk(k, dim=1)
            signed = contrib.gather(1, idx)
            tot_sq = (contrib * contrib).sum(dim=1).clamp_min(1e-12)
            kept_sq = (signed * signed).sum(dim=1)
            mlp_topk_out[li] = {
                "idx": idx.to(torch.int32).cpu(),
                "val": signed.to(torch.float16).cpu(),
                "energy_frac": (kept_sq / tot_sq).sqrt().to(torch.float16).cpu(),
            }

        handles.append(m.register_forward_hook(mhook))

    for aname, (am, nh) in attns.items():
        li = _layer_idx_from_name(aname)

        def ahook(mod, inp, out, aname=aname, li=li, nh=nh):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()  # [T, nh * hd]
            l2 = x.reshape(x.shape[0], nh, -1).norm(dim=2)
            head_out[aname] = {"layer": li, "n_heads": nh, "l2": l2.to(torch.float16).cpu()}

        handles.append(am.register_forward_hook(ahook))

    try:
        model(input_ids=ids.to(dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return CompactTrace(T, n_layers, hidden, resid_quant, group_size, top_k_mlp, resid_out, mlp_topk_out, head_out)


# ----------------------------------------------------------------------------------------------- dense trace_all diff
@torch.no_grad()
def diff_traces(
    model: nn.Module,
    ids_a: torch.Tensor | list[int],
    ids_b: torch.Tensor | list[int],
    *,
    wsi: Any | None = None,
    pos_a: int = -1,
    pos_b: int = -1,
    max_tokens: int = 64,
    top_k: int = 10,
) -> dict[str, Any]:
    """Full-fidelity X-ray comparing trace_all(ids_a) vs trace_all(ids_b) and linking shifts to WSI weights."""
    tr_a = trace_all(model, ids_a, max_tokens=max_tokens)
    tr_b = trace_all(model, ids_b, max_tokens=max_tokens)
    mods = dict(model.named_modules())

    def _at_pos(t: torch.Tensor, pos: int) -> torch.Tensor:
        return (t[0, pos] if t.dim() == 3 else t[pos]).float()

    # 1. Layer-by-layer residual curve across hidden_states
    hs_layers = sorted(
        int(k.split(".")[1]) for k in tr_a if k.startswith("hidden_states.") and k in tr_b
    )
    layer_curve = []
    peak_layer, peak_vec, peak_norm = None, None, -1.0
    for l in hs_layers:
        va = _at_pos(tr_a[f"hidden_states.{l}"], pos_a)
        vb = _at_pos(tr_b[f"hidden_states.{l}"], pos_b)
        dv = vb - va
        dn = float(dv.norm())
        rel = dn / max(float(va.norm()), 1e-12)
        cos = float(torch.nn.functional.cosine_similarity(va[None], vb[None]))
        layer_curve.append({"layer": l, "delta_norm": round(dn, 4), "rel_delta": round(rel, 5), "cosine": round(cos, 5)})
        if l > 0 and dn > peak_norm:
            peak_layer, peak_vec, peak_norm = l, dv, dn

    # 2. Causal MLP neuron shifts: delta_in[i] * ||W_down[:, i]||
    mlps = find_mlps(model)
    mlp_hits = []
    for g in mlps:
        key = f"{g.down}:in"
        if key not in tr_a or key not in tr_b:
            continue
        da = _at_pos(tr_a[key], pos_a)
        db = _at_pos(tr_b[key], pos_b)
        delta = db - da
        m = mods[g.down]
        w = m.weight.detach().float().cpu()
        wn = w.norm(dim=0) if isinstance(m, nn.Linear) else w.norm(dim=1)
        impact = delta * wn
        li = _layer_idx_from_name(g.down)
        for uid in impact.abs().topk(min(top_k, len(impact))).indices.tolist():
            mlp_hits.append({
                "layer": li, "module": g.down, "unit": int(uid),
                "act_a": round(float(da[uid]), 4), "act_b": round(float(db[uid]), 4),
                "resid_shift_norm": round(float(impact[uid].abs()), 4),
                "signed_shift": round(float(impact[uid]), 4),
            })
    mlp_hits.sort(key=lambda r: -r["resid_shift_norm"])
    top_mlp = mlp_hits[:top_k]
    if wsi is not None:
        for row in top_mlp:
            if row["layer"] is not None:
                try:
                    row["transduction_rule"] = wsi.explain_neuron(layer=row["layer"], unit=row["unit"])["transduction_rule"]
                except Exception:
                    pass

    # 3. Exact Attention / Linear-Attn Head output vector shift ||(x_b - x_a)_h @ W_O_h^T||
    hidden = peak_vec.numel() if peak_vec is not None else 0
    attns = _find_attn_out_projs(model, hidden) if hidden else {}
    head_hits = []
    for aname, (am, nh) in attns.items():
        key = f"{aname}:in"
        if key not in tr_a or key not in tr_b:
            continue
        da = _at_pos(tr_a[key], pos_a)
        db = _at_pos(tr_b[key], pos_b)
        dx = (db - da).reshape(nh, -1)
        w = am.weight.detach().float().cpu()  # Linear: [H, nh*hd], Conv1D: [nh*hd, H]
        w_heads = w.t().reshape(nh, -1, w.shape[0]) if isinstance(am, nn.Linear) else w.reshape(nh, -1, w.shape[1])
        # out_h = dx[h] @ w_heads[h] -> [nh, H]
        out_deltas = torch.einsum("hd,hdo->ho", dx, w_heads)
        norms = out_deltas.norm(dim=1)
        li = _layer_idx_from_name(aname)
        for hid in norms.topk(min(3, nh)).indices.tolist():
            row = {
                "layer": li, "module": aname, "head": int(hid),
                "out_delta_norm": round(float(norms[hid]), 4),
            }
            if li is not None and f"attentions.{li}" in tr_a and f"attentions.{li}" in tr_b:
                tb_att = tr_b[f"attentions.{li}"]
                ta_att = tr_a[f"attentions.{li}"]
                if tb_att.dim() == 4 and hid < tb_att.shape[1]:
                    att_a = ta_att[0, hid, pos_a].float()
                    att_b = tb_att[0, hid, pos_b].float()
                    row["attn_a_top_src_pos"] = int(att_a.argmax())
                    row["attn_a_top_weight"] = round(float(att_a.max()), 4)
                    row["attn_b_top_src_pos"] = int(att_b.argmax())
                    row["attn_b_top_weight"] = round(float(att_b.max()), 4)
                    shared = min(len(att_a), len(att_b))
                    if len(att_a) == len(att_b):
                        row["attn_max_shift_pos"] = int((att_b - att_a).abs().argmax())
                        row["attn_max_shift"] = round(float((att_b - att_a).abs().max()), 4)
                    elif shared > 0:
                        row["attn_prefix_l1_shift"] = round(float((att_b[:shared] - att_a[:shared]).abs().sum()), 4)
            head_hits.append(row)
    head_hits.sort(key=lambda r: -r["out_delta_norm"])

    # 4. Cache state deltas (KV at pos_a/pos_b even when T differs, + Gated DeltaNet recurrent/conv states)
    cache_shifts = []
    for k in sorted(set(tr_a) & set(tr_b)):
        if not k.startswith("cache."):
            continue
        ta, tb = tr_a[k].float(), tr_b[k].float()
        if ta.shape == tb.shape:
            cache_shifts.append({"key": k, "delta_norm": round(float((tb - ta).norm()), 4)})
        elif ta.dim() == 4 and tb.dim() == 4 and ta.shape[:2] == tb.shape[:2] and ta.shape[3] == tb.shape[3]:
            va, vb = ta[:, :, pos_a, :], tb[:, :, pos_b, :]
            cache_shifts.append({"key": f"{k}[pos]", "delta_norm": round(float((vb - va).norm()), 4)})
    cache_shifts.sort(key=lambda r: -r["delta_norm"])

    # 5. Optional static readout of peak residual shift via WSI
    vec_readout = None
    if wsi is not None and peak_vec is not None and peak_norm > 1e-6:
        try:
            ex = wsi.explain_vector(peak_vec / peak_vec.norm(), top_k=6)
            vec_readout = {"layer": peak_layer, "promoted": ex.get("promoted_tokens", []),
                           "suppressed": ex.get("suppressed_tokens", [])}
        except Exception:
            pass

    return {
        "len_a": int(tr_a[f"hidden_states.{hs_layers[0]}"].shape[1]) if hs_layers else None,
        "len_b": int(tr_b[f"hidden_states.{hs_layers[0]}"].shape[1]) if hs_layers else None,
        "pos_a": pos_a, "pos_b": pos_b,
        "peak_residual_layer": peak_layer, "peak_residual_delta_norm": round(peak_norm, 4),
        "layer_curve": layer_curve,
        "top_mlp_contributors": top_mlp,
        "top_attention_heads": head_hits[:top_k],
        "top_cache_deltas": cache_shifts[:top_k],
        "peak_vector_readout": vec_readout,
    }
