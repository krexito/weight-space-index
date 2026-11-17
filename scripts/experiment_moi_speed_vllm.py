"""Does Compensated Selective MoI actually run faster? Real sliced checkpoints served by vLLM.

The LFM3 run (benchmarks/moi_task_index_lfm3) showed the quality side: a compensated math index keeps 79% GSM8K at
70% of MLP neurons. Its speed was measured with HF generate, which is overhead-bound at that size, so no speedup was
visible. vLLM has no LFM3 support yet, so this runs on Qwen3.5-0.8B, which vLLM serves.

  export   calibrate on N GSM8K-train solution tokens (default 20k), pick each layer's top neurons by `energy`
           (sqrt E[h_i^2] * ||W_down[:, i]||, same share in every layer), refit the kept down_proj columns by ridge
           least squares on the co-activation Gram (W_S = W G[:,S] (G_SS + lam I)^-1), and WRITE a real checkpoint
           with smaller gate/up/down matrices and `intermediate_size` = kept neurons. The unused MTP head is sliced
           to the same width (by down_proj column norm) so every shape matches the config.
           --neurons 2560,2048,1792,1024 keeps exact counts instead (multiples of 128: the first T4 run showed
           2509 and 1075 neurons decoding SLOWER than 1792, because partial matmul tiles waste work).
  bench    one checkpoint in vLLM: decode tok/s at batch 1 (the on-device case: weight reads dominate), throughput
           at batch 64, GSM8K accuracy (200 test problems, greedy, thinking off), checkpoint size.
  report   merge the bench files.

    python experiment_moi_speed_vllm.py export --model Qwen/Qwen3.5-0.8B --out-dir /tmp/ckpt --data-dir /tmp/data
    python experiment_moi_speed_vllm.py bench --model-dir /tmp/ckpt/energy_recon_0.5 --tag 0.5 --res-dir R
    python experiment_moi_speed_vllm.py report --res-dir R
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import time
from pathlib import Path

HUB_FILES = {
    "gsm8k/test.parquet": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    "gsm8k/train.parquet": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
    "mbpp/train.parquet": ("google-research-datasets/mbpp", "full/train-00000-of-00001.parquet"),
    "mbpp/validation.parquet": ("google-research-datasets/mbpp", "full/validation-00000-of-00001.parquet"),
    "mbpp/test.parquet": ("google-research-datasets/mbpp", "full/test-00000-of-00001.parquet"),
    "mbpp/prompt.parquet": ("google-research-datasets/mbpp", "full/prompt-00000-of-00001.parquet"),
    "mmlu/auxiliary_train.parquet": ("cais/mmlu", "all/auxiliary_train-00000-of-00001.parquet"),
    "wikitext/train.parquet": ("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet"),
    "humaneval/test.parquet": ("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"),
    "mmlu/test.parquet": ("cais/mmlu", "all/test-00000-of-00001.parquet"),
    "mmlu/validation.parquet": ("cais/mmlu", "all/validation-00000-of-00001.parquet"),
}
PROMPT_SUFFIX = "\n\nSolve step by step. End with: The answer is <number>."


def need(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    repo, fn = HUB_FILES[name]
    got = Path(hf_hub_download(repo, fn, repo_type="dataset", local_dir=data_dir / "_hub"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(got.read_bytes())
    return path


def clean_solution(ans: str) -> str:
    body, final = ans.split("####")
    return re.sub(r"<<[^>]*>>", "", body).strip() + f"\nThe answer is {final.strip()}."


def extract_number(text: str) -> str | None:
    text = text.replace(",", "")
    for pat in (r"\\boxed\{([^}]*)\}", r"answer is[:\s]*\$?\s*(-?\d+(?:\.\d+)?)", r"####\s*(-?\d+(?:\.\d+)?)"):
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            nums = re.findall(r"-?\d+(?:\.\d+)?", m[-1])
            if nums:
                return nums[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def same_number(a: str | None, b: str) -> bool:
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def calib_sequences(source: str, tok, data_dir: Path, n_tokens: int) -> list[list[int]]:
    """Token sequences for calibrating a task index ('zone'), until n_tokens are reached.
    gsm8k: GSM8K-train question + worked solution (chat format). mbpp: MBPP-train task + reference Python solution
    (chat format; train first, then the validation / test / prompt splits when more tokens are asked for: the code zone
    is evaluated on HumanEval, a different dataset). wikitext: plain WikiText-2 train paragraphs (general text, no chat
    template). mmlu_aux: MMLU auxiliary_train (ARC, OpenBookQA, RACE... in MMLU's multiple-choice format, none of it
    from the MMLU test), shuffled, asked and answered in chat format exactly like the MMLU evaluation.
    mix: a third of n_tokens from each of gsm8k, mbpp and mmlu_aux, interleaved (one index for all tasks)."""
    import pandas as pd
    import numpy as np

    def chat(u, a_):
        return tok(tok.apply_chat_template([{"role": "user", "content": u}, {"role": "assistant", "content": a_}],
                                           tokenize=False, enable_thinking=False), add_special_tokens=False).input_ids

    if source == "gsm8k":
        d = pd.read_parquet(need(data_dir, "gsm8k/train.parquet"))
        gen = (chat(q, clean_solution(ans)) for q, ans in zip(d["question"], d["answer"]))
    elif source == "mbpp":
        d = pd.concat([pd.read_parquet(need(data_dir, f"mbpp/{s}.parquet")) for s in ("train", "validation", "test", "prompt")],
                      ignore_index=True)
        gen = (chat(f"{t}\nYour code should pass this test: {tl[0]}",
                    "```python\n" + c.replace("\r\n", "\n").replace("\t", "    ").strip() + "\n```")
               for t, c, tl in zip(d["text"], d["code"], d["test_list"]))
    elif source == "wikitext":
        d = pd.read_parquet(need(data_dir, "wikitext/train.parquet"))
        paras = [t.strip() for t in d["text"] if len(t.strip()) > 200 and not t.strip().startswith("=")]
        gen = (tok(p_, add_special_tokens=False).input_ids[:1024] for p_ in paras)
    elif source == "mix":  # one index for everything: a third of the tokens from each task's calibration text
        parts = [calib_sequences(s, tok, data_dir, n_tokens // 3) for s in ("gsm8k", "mbpp", "mmlu_aux")]
        out = [ids for grp in zip(*[p_ + [None] * (max(map(len, parts)) - len(p_)) for p_ in parts]) for ids in grp if ids]
        return out
    elif source == "mmlu_aux":
        d = pd.read_parquet(need(data_dir, "mmlu/auxiliary_train.parquet"))
        d = d[d["question"].str.len() < 1000]  # mostly short science / commonsense questions, some reading passages
        d = d.iloc[np.random.default_rng(0).permutation(len(d))]
        gen = (chat(q.strip() + "\n" + "\n".join(f"{L}. {c}" for L, c in zip("ABCD", ch)) + "\nAnswer with only the letter.",
                    "ABCD"[int(ans)])
               for q, ch, ans in zip(d["question"], d["choices"], d["answer"]))
    else:
        raise SystemExit(f"unknown calibration source {source}")
    out, n = [], 0
    for ids in gen:
        out.append(ids)
        n += len(ids)
        if n >= n_tokens:
            break
    return out


def local_dir(model: str) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model))


def compensate_layer(
    W,
    G,
    sq,
    k: int,
    ridge: float = 1e-2,
    ridge_target: str = "zero",
    down_mode: str = "full",
    refit_frac: float = 0.25,
    core_indices=None,
    lowrank_r: int = 128,
):
    """Select top-k neurons by activation energy and solve down_proj compensation.

    Returns (ix, Ws, refit_mask, metrics, lowrank_factors) on W's device:
      - ridge_target='zero': legacy objective min ||W x - Ws x_s||^2 + lam ||Ws||^2
      - ridge_target='delta': shrinks delta = Ws - W_s toward 0 (leaves low-variance weights intact)
      - ridge_target='delta_diag': per-neuron variance-scaled delta ridge (lam_j = ridge * G_jj)
      - down_mode='shared_sparse': updates only top refit_frac columns by ||delta||*sqrt(G_jj), keeps rest = W
      - down_mode='freeze_core': freezes columns in core_indices to W, updates only S \\ core
      - down_mode='lowrank': stores Ws = W[:, ix] + U_r @ V_r^T (rank lowrank_r) for compact delta transfer
    """
    import torch

    dev = W.device
    U_dim = W.shape[1]
    ix = (sq.sqrt() * W.norm(dim=0)).topk(k).indices.sort().values
    keep_mask = torch.zeros(U_dim, dtype=torch.bool, device=dev)
    keep_mask[ix] = True
    dropped = (~keep_mask).nonzero(as_tuple=True)[0]

    def _solve_cols(cols_idx, target_mode: str):
        if len(cols_idx) == 0:
            return W[:, cols_idx].clone()
        Gmm = G[cols_idx][:, cols_idx]
        if target_mode == "zero":
            lam = ridge * Gmm.diagonal().mean().clamp_min(1e-12)
            rhs = (W @ G[:, cols_idx]).T
            return torch.linalg.solve(Gmm + lam * torch.eye(len(cols_idx), device=dev, dtype=torch.float64), rhs).T
        if len(dropped) == 0:
            return W[:, cols_idx].clone()
        rhs = (W[:, dropped] @ G[dropped][:, cols_idx]).T  # [m, H]
        if target_mode == "delta":
            lam = ridge * Gmm.diagonal().mean().clamp_min(1e-12)
            reg = lam * torch.eye(len(cols_idx), device=dev, dtype=torch.float64)
        elif target_mode == "delta_diag":
            diag = Gmm.diagonal().clamp_min(1e-12)
            reg = torch.diag(ridge * diag + 1e-8 * diag.mean())
        else:
            raise ValueError(f"unknown ridge_target: {target_mode}")
        delta_T = torch.linalg.solve(Gmm + reg, rhs)
        return W[:, cols_idx] + delta_T.T

    refit_mask = torch.ones(k, dtype=torch.bool, device=dev)
    lr_factors = None
    if down_mode == "full":
        Ws = _solve_cols(ix, ridge_target)
    elif down_mode == "shared_sparse":
        eff_target = "delta" if ridge_target == "zero" else ridge_target
        Ws_full = _solve_cols(ix, eff_target)
        delta_full = Ws_full - W[:, ix]
        impact = delta_full.norm(dim=0) * sq[ix].clamp_min(0).sqrt()
        m = max(1, min(k, round(refit_frac * k)))
        sub_pos = impact.topk(m).indices.sort().values
        refit_mask.zero_()
        refit_mask[sub_pos] = True
        Ws = W[:, ix].clone()
        Ws[:, sub_pos] = _solve_cols(ix[sub_pos], eff_target)
    elif down_mode == "freeze_core":
        eff_target = "delta" if ridge_target == "zero" else ridge_target
        core_set = torch.zeros(U_dim, dtype=torch.bool, device=dev)
        if core_indices is not None and len(core_indices) > 0:
            core_set[torch.as_tensor(core_indices, device=dev, dtype=torch.long)] = True
        refit_mask = ~core_set[ix]
        Ws = W[:, ix].clone()
        unlock_idx = ix[refit_mask]
        if len(unlock_idx) > 0:
            Ws[:, refit_mask] = _solve_cols(unlock_idx, eff_target)
    elif down_mode == "lowrank":
        eff_target = "delta" if ridge_target == "zero" else ridge_target
        Ws_full = _solve_cols(ix, eff_target)
        delta_full = Ws_full - W[:, ix]
        U_svd, S_svd, Vh_svd = torch.linalg.svd(delta_full, full_matrices=False)
        r = min(lowrank_r, len(S_svd))
        U_r = (U_svd[:, :r] * S_svd[:r].unsqueeze(0)).contiguous()
        V_r = Vh_svd[:r, :].contiguous()
        Ws = W[:, ix] + U_r @ V_r
        lr_factors = (U_r, V_r)
    else:
        raise ValueError(f"unknown down_mode: {down_mode}")

    # Reconstruction error E[||W x - Ws x_s||^2] / E[||W x||^2] vs uncompensated truncation
    tot = torch.trace(W @ G @ W.T).clamp_min(1e-12)
    recon = (tot - 2 * torch.trace(Ws @ G[ix] @ W.T) + torch.trace(Ws @ G[ix][:, ix] @ Ws.T)).clamp_min(0)
    if len(dropped) > 0:
        uncomp = torch.trace(W[:, dropped] @ G[dropped][:, dropped] @ W[:, dropped].T).clamp_min(0)
    else:
        uncomp = torch.tensor(0.0, device=dev, dtype=torch.float64)
    metrics = {
        "recon_rel_l2": float((recon / tot).sqrt()),
        "uncomp_rel_l2": float((uncomp / tot).sqrt()),
        "refit_cols": int(refit_mask.sum()),
    }
    return ix, Ws, refit_mask, metrics, lr_factors


# ---------------------------------------------------------------------------------------------------------- export
def cmd_export(a) -> None:
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = local_dir(a.model)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    tok = AutoTokenizer.from_pretrained(src)
    def sync() -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize()

    cdt = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
    if a.calib_dtype != "auto":
        cdt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.calib_dtype]

    timing: dict = {}
    grams_path = Path(a.grams_cache) if a.grams_cache else None
    downs: dict[int, str] = {}
    for f_ in sorted(src.glob("*.safetensors")):
        with safe_open(str(f_), "pt") as s_:
            for kk in s_.keys():
                m = re.search(r"layers\.(\d+)\.mlp\.down_proj\.weight$", kk)
                if m and "mtp" not in kk:
                    downs[int(m.group(1))] = kk[:-len(".weight")]
    L = len(downs)

    files = sorted(src.glob("*.safetensors"))
    keys = {}
    for f in files:
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                keys[k] = f

    def ckey(li: int, proj: str, mtp: bool = False) -> str:
        pre = "mtp." if mtp else ""
        cands = [k for k in keys if k.endswith(f"layers.{li}.mlp.{proj}.weight") and k.startswith("mtp.") == mtp]
        if len(cands) != 1:
            raise SystemExit(f"cannot find a unique {pre}layer {li} {proj} key: {cands}")
        return cands[0]

    with safe_open(str(keys[ckey(sorted(downs)[0], "down_proj")]), "pt") as s_:
        U = s_.get_slice(ckey(sorted(downs)[0], "down_proj")).get_shape()[1]

    if grams_path and grams_path.exists():
        t0 = time.perf_counter()
        cached_g = torch.load(str(grams_path), map_location="cpu", weights_only=False)
        Gcpu, sqd, ntok = cached_g["G"], cached_g["sq"], cached_g["ntok"]
        timing["load_model_s"] = 0.0
        timing["calibration_s"] = round(time.perf_counter() - t0, 2)
        timing["calibration_passes"] = 0
        timing["from_grams_cache"] = True
        print(f"[cache] loaded {ntok:,} tokens Grams from {grams_path} in {timing['calibration_s']:.2f}s", flush=True)
    else:
        t_load = time.perf_counter()
        # bf16 where the GPU supports it (the checkpoint's own dtype); fp32 otherwise. Never fp16: these models have large
        # activations that overflow it. Statistics are always accumulated in fp32/fp64.
        model = AutoModelForCausalLM.from_pretrained(src, dtype=cdt).to(dev).eval()
        sync()
        timing["load_model_s"] = round(time.perf_counter() - t_load, 2)
        mods = dict(model.named_modules())
        # module names can differ from the checkpoint keys (Qwen3.5: "model.language_model.layers..." on disk), so the
        # hooks are found by layer index among the loaded modules
        mod_downs = {}
        for name in mods:
            m = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
            if m and "mtp" not in name:
                mod_downs[int(m.group(1))] = name
        if sorted(mod_downs) != sorted(downs):
            raise SystemExit(f"down_proj layers differ between the checkpoint ({len(downs)}) and the model ({len(mod_downs)})")
        print(f"{L} MLPs x {U} neurons", flush=True)

        calib = calib_sequences(a.calib_source, tok, Path(a.data_dir), a.calib_tokens)

        # The Gram matrices of every layer may not fit next to the model (Qwen3.5-4B: 32 x 9216^2 fp32 = 10.9 GB), so the
        # layers are calibrated in groups that fit in free GPU memory; each finished group's Grams go to CPU RAM.
        order = sorted(downs)
        if a.layers_per_pass:
            per = a.layers_per_pass
        elif dev.type == "cuda":
            per = int((torch.cuda.mem_get_info()[0] - 6 * 2**30) // (U * U * 4))
        else:
            per = L
        per = max(1, min(per, L))
        groups = [order[i : i + per] for i in range(0, L, per)]
        Gcpu, sqd = {}, {}
        t0, ntok = time.perf_counter(), 0
        for gi, grp in enumerate(groups):
            st = {li: {"sq": torch.zeros(U, device=dev, dtype=torch.float64),
                       "G": torch.zeros(U, U, device=dev, dtype=torch.float32)} for li in grp}

            def hook(li):
                def f(mod, inp):
                    x = inp[0].detach().reshape(-1, U).float()
                    st[li]["sq"] += (x * x).sum(0).double()
                    st[li]["G"] += x.T @ x
                return f

            hs = [mods[mod_downs[li]].register_forward_pre_hook(hook(li)) for li in grp]
            n_pass = 0
            with torch.no_grad():
                for s in calib:
                    model(input_ids=torch.tensor([s], device=dev), use_cache=False)
                    n_pass += len(s)
            for h in hs:
                h.remove()
            ntok = n_pass
            for li in grp:
                Gcpu[li], sqd[li] = st[li]["G"].cpu(), st[li]["sq"].cpu()
            del st
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  calibration pass {gi + 1}/{len(groups)}: layers {grp[0]}-{grp[-1]}", flush=True)
        sync()
        timing["calibration_s"] = round(time.perf_counter() - t0, 2)
        timing["calibration_passes"] = len(groups)
        print(f"[time] load model {timing['load_model_s']:.1f}s | calibration: {ntok:,} tokens, {len(groups)} pass(es) in "
              f"{timing['calibration_s']:.1f}s", flush=True)
        del model, mods, hs
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if grams_path:
            grams_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"G": Gcpu, "sq": sqd, "ntok": ntok}, str(grams_path))
            print(f"[cache] saved Grams to {grams_path}", flush=True)

    # The refit only needs each layer's down_proj weight: keep those on CPU so the fp64 solves have room.
    downW = {}
    for li in sorted(downs):
        dk = ckey(li, "down_proj")
        with safe_open(str(keys[dk]), "pt") as s:
            downW[li] = s.get_tensor(dk).detach().clone()

    core_kept = dict(np.load(a.core_file)) if a.core_file else None
    mtp_layers = sorted({int(m.group(1)) for k in keys for m in [re.match(r"mtp\.layers\.(\d+)\.mlp\.down_proj", k)] if m})
    summary = {"model": a.model, "calib_source": a.calib_source, "calib_tokens": ntok, "calib_dtype": str(cdt), "layers": L,
               "neurons_per_layer": U, "ridge": a.ridge, "ridge_target": a.ridge_target, "down_mode": a.down_mode,
               "refit_frac": a.refit_frac if a.down_mode == "shared_sparse" else None,
               "lowrank_r": a.lowrank_r if a.down_mode == "lowrank" else None,
               "budgets": {}}
    if a.neurons:  # exact kept-neuron counts, e.g. multiples of 128 so the GPU matmul tiles are full
        targets = [(f"k{int(x)}", int(x)) for x in a.neurons.split(",")]
    else:
        targets = [(f"{float(x):g}", round(float(x) * U)) for x in a.budgets.split(",")]
    for name, k in targets:
        t_rec, t_slice = time.perf_counter(), 0.0
        new, kept, refit_masks, lr_tens = {}, {}, {}, {}
        layer_metrics = []
        for li in sorted(downs):
            W = downW[li].to(dev).double()                                # [H, U]
            sq, G = sqd[li].to(dev) / ntok, Gcpu[li].to(dev).double() / ntok
            c_idx = core_kept[f"layer{li}"] if core_kept and f"layer{li}" in core_kept else None
            ix, Ws, rmask, lm, lr_fac = compensate_layer(
                W, G, sq, k, ridge=a.ridge, ridge_target=a.ridge_target, down_mode=a.down_mode,
                refit_frac=a.refit_frac, core_indices=c_idx, lowrank_r=a.lowrank_r,
            )
            layer_metrics.append(lm)
            sync()
            t_s = time.perf_counter()
            ixc = ix.cpu()
            kept[li] = ixc.numpy()
            refit_masks[li] = rmask.cpu().numpy()
            with safe_open(str(keys[ckey(li, "gate_proj")]), "pt") as s:
                gate = s.get_tensor(ckey(li, "gate_proj"))
            with safe_open(str(keys[ckey(li, "up_proj")]), "pt") as s:
                up = s.get_tensor(ckey(li, "up_proj"))
            dt = gate.dtype
            new[ckey(li, "gate_proj")] = gate[ixc].contiguous()
            new[ckey(li, "up_proj")] = up[ixc].contiguous()
            new[ckey(li, "down_proj")] = Ws.to("cpu", dt).contiguous()
            if lr_fac is not None:
                lr_tens[f"layer{li}.u"] = lr_fac[0].to("cpu", dt).contiguous()
                lr_tens[f"layer{li}.v"] = lr_fac[1].to("cpu", dt).contiguous()
            t_slice += time.perf_counter() - t_s
        for li in mtp_layers:  # never used for plain decoding; slice so the shapes match intermediate_size
            with safe_open(str(keys[ckey(li, "down_proj", True)]), "pt") as s:
                dn = s.get_tensor(ckey(li, "down_proj", True))
            ix = dn.float().norm(dim=0).topk(k).indices.sort().values
            new[ckey(li, "down_proj", True)] = dn[:, ix].contiguous()
            for p in ("gate_proj", "up_proj"):
                with safe_open(str(keys[ckey(li, p, True)]), "pt") as s:
                    new[ckey(li, p, True)] = s.get_tensor(ckey(li, p, True))[ix].contiguous()

        recon_s = time.perf_counter() - t_rec - t_slice  # selection + ridge refit only (reading weights excluded)
        t_w = time.perf_counter()
        dst = out / f"energy_recon_{name}"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in src.iterdir():
            if f.is_file() and f.suffix != ".safetensors" and f.name != "config.json":
                shutil.copy2(f, dst / f.name)
        cfg = json.loads((src / "config.json").read_text())
        for c in (cfg, cfg.get("text_config", {})):
            if "intermediate_size" in c:
                c["intermediate_size"] = k
        (dst / "config.json").write_text(json.dumps(cfg, indent=2))
        np.savez(dst / "kept_neurons.npz", **{f"layer{li}": v for li, v in kept.items()})
        np.savez(dst / "refit_mask.npz", **{f"layer{li}": v for li, v in refit_masks.items()})
        if lr_tens:
            save_file(lr_tens, str(dst / "delta_lowrank.safetensors"))
        for f in files:
            with safe_open(str(f), "pt") as s:
                meta = s.metadata() or {}
                tens = {kk: new[kk] if kk in new else s.get_tensor(kk) for kk in s.keys()}
            save_file(tens, str(dst / f.name), metadata={**meta, "format": "pt"})
            del tens
        write_s = time.perf_counter() - t_w + t_slice
        mean_recon = float(np.mean([m["recon_rel_l2"] for m in layer_metrics]))
        mean_uncomp = float(np.mean([m["uncomp_rel_l2"] for m in layer_metrics]))
        mean_refit = float(np.mean([m["refit_cols"] for m in layer_metrics]))
        lr_bytes = (dst / "delta_lowrank.safetensors").stat().st_size if lr_tens else 0
        summary["budgets"][name] = {
            "kept_neurons": k, "dir": str(dst),
            "mean_recon_rel_l2": round(mean_recon, 5), "mean_uncomp_rel_l2": round(mean_uncomp, 5),
            "mean_refit_cols": round(mean_refit, 1), "lowrank_delta_bytes": lr_bytes,
            "time_s": {"select_and_compensate": round(recon_s, 2), "slice_and_write_checkpoint": round(write_s, 2)},
            "bytes": sum(f.stat().st_size for f in dst.glob("*.safetensors") if f.name != "delta_lowrank.safetensors"),
        }
        print(f"[time] {name}: select+compensate {recon_s:.1f}s | slice+write {write_s:.1f}s | "
              f"rel_L2 {mean_uncomp:.4f} -> {mean_recon:.4f} | refit_cols {mean_refit:.0f}/{k}")
        print(f"wrote {dst}  ({k}/{U} neurons per layer, {summary['budgets'][name]['bytes'] / 2**30:.2f} GiB)",
              flush=True)
    summary["timing"] = timing
    summary["device"] = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    summary["full_bytes"] = sum(f.stat().st_size for f in files)
    summary["full_dir"] = str(src)
    (out / "export.json").write_text(json.dumps(summary, indent=2))


# ----------------------------------------------------------------------------------------------------------- bench
def cmd_bench(a) -> None:
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    md = local_dir(a.model_dir)
    tok = AutoTokenizer.from_pretrained(md)
    dtype = a.dtype
    if dtype == "auto" and torch.cuda.get_device_capability()[0] < 8:
        dtype = "half"  # T4 / P100: no bf16
    kw = dict(model=str(md), dtype=dtype, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=0)
    llm = None
    for extra in ({"limit_mm_per_prompt": {"image": 0, "video": 0}}, {}):
        try:
            llm = LLM(**kw, **extra)
            break
        except Exception as e:  # noqa: BLE001
            print("vLLM start failed with", extra, "->", repr(e)[:300], flush=True)
    if llm is None:
        raise SystemExit("vLLM could not start")

    test = pd.read_parquet(need(Path(a.data_dir), "gsm8k/test.parquet")).iloc[: a.n_problems]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + PROMPT_SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for q in test["question"]]

    def timed(ps: list[str], n_new: int) -> float:
        sp = SamplingParams(temperature=0.0, max_tokens=n_new, ignore_eos=True)
        t = time.perf_counter()
        outs = llm.generate(ps, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        return sum(len(o.outputs[0].token_ids) for o in outs) / dt

    timed(prompts[:4], 32)  # warmup (CUDA graphs, kernels)
    b1 = [timed(prompts[i : i + 1], a.decode_tokens) for i in range(a.repeats)]
    thr = [timed(prompts[:a.batch], a.batch_tokens) for _ in range(2)]
    print(f"[{a.tag}] batch 1: {statistics.median(b1):.1f} tok/s   batch {a.batch}: {max(thr):.0f} tok/s", flush=True)

    t = time.perf_counter()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_new), use_tqdm=False)
    gen_s = time.perf_counter() - t
    ok, samples, ntoks = 0, [], 0
    for o, gold in zip(outs, test["answer"]):
        g = gold.split("####")[-1].strip().replace(",", "")
        txt = o.outputs[0].text
        pred = extract_number(txt.split("</think>")[-1])
        ok += same_number(pred, g)
        ntoks += len(o.outputs[0].token_ids)
        if len(samples) < 3:
            samples.append({"gold": g, "pred": pred, "text": txt[-300:]})
    sizes = sum(f.stat().st_size for f in md.glob("*.safetensors"))
    cfg = json.loads((md / "config.json").read_text())
    res = {"tag": a.tag, "model_dir": str(md), "dtype": dtype, "gpu": torch.cuda.get_device_name(),
           "intermediate_size": cfg.get("text_config", cfg).get("intermediate_size"),
           "checkpoint_bytes": sizes,
           "decode_tok_s_batch1": round(statistics.median(b1), 2), "decode_tok_s_batch1_runs": [round(x, 2) for x in b1],
           f"throughput_tok_s_batch{a.batch}": round(max(thr), 1),
           "gsm8k": {"accuracy": ok / len(prompts), "correct": ok, "n": len(prompts), "gen_tokens": ntoks,
                     "wall_s": round(gen_s, 1), "samples": samples}}
    rd = Path(a.res_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"bench_{a.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[{a.tag}] GSM8K {ok}/{len(prompts)} = {ok / len(prompts):.1%}  ({gen_s:.0f}s)", flush=True)


def cmd_report(a) -> None:
    rd = Path(a.res_dir)
    rows = {}
    for f in sorted(rd.glob("bench_*.json")):
        r = json.loads(f.read_text())
        rows[r["tag"]] = r
    exp = Path(a.export_json)
    res = {"export": json.loads(exp.read_text()) if exp.exists() else None, "arms": rows}
    if res["export"] and "timing" in res["export"]:
        t = res["export"]["timing"]
        print(f"\nBuild time on {res['export'].get('device')}: load {t['load_model_s']:.1f}s, calibration "
              f"({res['export']['calib_tokens']:,} tokens) {t['calibration_s']:.1f}s")
        for n, bgt in res["export"]["budgets"].items():
            ts = bgt.get("time_s", {})
            print(f"  {n}: select+compensate {ts.get('select_and_compensate', 0):.1f}s, "
                  f"slice+write {ts.get('slice_and_write_checkpoint', 0):.1f}s")
    if "full" in rows:
        f = rows["full"]
        bk = next(k for k in f if k.startswith("throughput_tok_s_batch"))
        print(f"\n{'arm':6s} {'neurons':>8s} {'size':>8s} {'GSM8K':>7s} {'batch1':>12s} {'batch-N':>12s}")
        for t, r in rows.items():
            print(f"{t:6s} {r['intermediate_size']:8d} {r['checkpoint_bytes'] / 2**30:7.2f}G "
                  f"{r['gsm8k']['accuracy']:7.1%} {r['decode_tok_s_batch1']:7.1f} ({r['decode_tok_s_batch1'] / f['decode_tok_s_batch1']:.2f}x)"
                  f" {r[bk]:7.0f} ({r[bk] / f[bk]:.2f}x)")
        res["speedup_vs_full"] = {t: {"batch1": round(r["decode_tok_s_batch1"] / f["decode_tok_s_batch1"], 3),
                                      "batchN": round(r[bk] / f[bk], 3),
                                      "size": round(r["checkpoint_bytes"] / f["checkpoint_bytes"], 3)}
                                  for t, r in rows.items()}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("Saved", a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--data-dir", required=True)
    e.add_argument("--budgets", default="0.7,0.5,0.3")
    e.add_argument("--neurons", help="exact kept neurons per layer, comma-separated (overrides --budgets)")
    e.add_argument("--calib-tokens", type=int, default=20_000)
    e.add_argument("--calib-source", default="gsm8k", choices=["gsm8k", "mbpp", "wikitext", "mmlu_aux", "mix"],
                   help="text the index is calibrated on (one 'zone' per task)")
    e.add_argument("--ridge", type=float, default=1e-2)
    e.add_argument("--ridge-target", default="zero", choices=["zero", "delta", "delta_diag"],
                   help="zero = legacy shrink Ws->0; delta = shrink Ws-W_base->0; delta_diag = variance-scaled delta")
    e.add_argument("--down-mode", default="full", choices=["full", "shared_sparse", "freeze_core", "lowrank"],
                   help="full = refit all kept cols; shared_sparse = refit top frac; freeze_core = lock core to W_base; lowrank = SVD delta")
    e.add_argument("--refit-frac", type=float, default=0.25, help="shared_sparse: fraction of kept columns refitted")
    e.add_argument("--core-file", help="freeze_core: path to kept_neurons.npz whose shared columns stay at W_base")
    e.add_argument("--lowrank-r", type=int, default=128, help="lowrank: SVD rank for down_proj delta")
    e.add_argument("--grams-cache", help="optional .pt file to save/reuse calibration Grams")
    e.add_argument("--device", default="cuda")
    e.add_argument("--calib-dtype", default="auto", choices=["auto", "bf16", "fp32"])
    e.add_argument("--layers-per-pass", type=int, default=0, help="0 = as many as fit in free GPU memory")
    b = sub.add_parser("bench")
    b.add_argument("--model-dir", required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--res-dir", required=True)
    b.add_argument("--data-dir", required=True)
    b.add_argument("--n-problems", type=int, default=200)
    b.add_argument("--max-new", type=int, default=1024)
    b.add_argument("--decode-tokens", type=int, default=512)
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--batch", type=int, default=64)
    b.add_argument("--batch-tokens", type=int, default=256)
    b.add_argument("--dtype", default="auto")
    b.add_argument("--max-model-len", type=int, default=4096)
    b.add_argument("--gpu-mem", type=float, default=0.85)
    r = sub.add_parser("report")
    r.add_argument("--res-dir", required=True)
    r.add_argument("--export-json", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"export": cmd_export, "bench": cmd_bench, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
