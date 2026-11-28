"""Physical Grafting Limit Stress Test on Real LoopedWSP Circuits (Fast NumPy CPU).

Evaluates 3 fast, size-general LoopedWSP networks (`maze`, `hanoi`, `graph`) under 4
perturbations simulating real pretrained LLM residual-stream grafting:
  1. Precision cliff: fp64 (baseline), fp32, fp16, bf16_weights_only, bf16_acts_only, bf16_both.
  2. Gaussian noise cliff: sigma in {0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} in `init_only` and `per_loop` modes.
  3. RMSNorm injection: unit-gain RMSNorm applied `pre_block` and `post_block` inside each loop.
  4. High-dimensional linear projection interface (D=1024, matching Qwen3.5-0.8B): closed-form
     ridge projector fit on calibration embeddings across distractor levels eta, measuring RMSE,
     SNR (dB), and held-out task accuracy + fixed-point survival.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi.coproc import verify_hanoi_moves, verify_maze_solution


# ------------------------------------------------------------------ Deterministic Dataset Generators

def _make_seeded_maze(n: int, seed: int) -> str:
    rng = random.Random(seed)
    g, queue = [["#"] * n for _ in range(n)], [(1, 1)]
    g[1][1] = " "
    while queue:
        r, c = queue[-1]
        nxt = [
            (r + a, c + b)
            for a, b in ((2, 0), (-2, 0), (0, 2), (0, -2))
            if 0 < r + a < n - 1 and 0 < c + b < n - 1 and g[r + a][c + b] == "#"
        ]
        if not nxt:
            queue.pop()
            continue
        r2, c2 = rng.choice(nxt)
        g[(r + r2) // 2][(c + c2) // 2] = g[r2][c2] = " "
        queue.append((r2, c2))
    for _ in range(n):
        g[rng.randrange(1, n - 1)][rng.randrange(1, n - 1)] = " "
    open_cells = [(r, c) for r in range(n) for c in range(n) if g[r][c] == " "]
    (a, b), (c, d) = rng.sample(open_cells, 2)
    g[a][b], g[c][d] = "S", "G"
    return "\n".join("".join(row) for row in g)


def _make_seeded_graph(n_nodes: int, seed: int) -> str:
    rng = random.Random(seed)
    ids = rng.sample(range(64), 64)
    edges = set()
    while len(edges) < min(round(1.5 * n_nodes), n_nodes * (n_nodes - 1)):
        edges.add(tuple(rng.sample(ids[:n_nodes], 2)))
    a, b = rng.sample(ids[:n_nodes], 2)
    return " ".join(f"{u}>{v}" for u, v in sorted(edges)) + f" ? {a} {b}"


def _graph_truth(toks: list[list[int]], io: dict[str, Any]) -> str:
    a, b = toks[-1][1:]
    edges = [t[1:] for t in toks if t[0] == io["kind"]["edge"]]
    dist, queue = {a: 0}, [a]
    for u in queue:
        for x, y in edges:
            if x == u and y not in dist:
                dist[y] = dist[u] + 1
                queue.append(y)
    return str(dist[b]) if b in dist else "unreachable"


# ------------------------------------------------------------------ Exact Precision Helpers

def _to_bf16_np(arr: np.ndarray, out_dtype=np.float64) -> np.ndarray:
    """Bit-exact IEEE-754 round-to-nearest-even float32->bfloat16 quantization in NumPy."""
    f32 = np.asarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    is_nan = np.isnan(f32)
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    rounded = np.where(
        is_nan,
        u32 | np.uint32(0x0040_0000),
        u32 + np.uint32(0x7FFF) + lsb,
    ) & np.uint32(0xFFFF_0000)
    return rounded.view(np.float32).astype(out_dtype)


def _verify_bf16_matches_torch() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([
        rng.standard_normal(256).astype(np.float32),
        np.array([0.0, -0.0, 1.0, -1.0, 1e4, 1e-4, np.inf, -np.inf], dtype=np.float32),
    ])
    np_q = _to_bf16_np(x, np.float32)
    th_q = torch.from_numpy(x.copy()).to(torch.bfloat16).float().numpy()
    assert np.array_equal(np_q, th_q, equal_nan=True), "bf16 quantization mismatch vs PyTorch"


def _quantize(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "fp64":
        return np.asarray(arr, dtype=np.float64)
    if mode == "fp32":
        return np.asarray(arr, dtype=np.float32).astype(np.float64)
    if mode == "fp16":
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(arr, dtype=np.float16).astype(np.float64)
    if mode == "bf16":
        return _to_bf16_np(arr, np.float64)
    raise ValueError(f"Unknown precision mode: {mode}")


# ------------------------------------------------------------------ Network Loader & Fast Runner

def _load_network_np(net_dir: Path, name: str) -> dict[str, Any]:
    cfg = json.loads((net_dir / f"{name}.json").read_text(encoding="utf-8"))
    with np.load(net_dir / f"{name}.npz", allow_pickle=False) as z:
        def dense(k: str) -> np.ndarray:
            shape = tuple(int(s) for s in z[k + ".shape"])
            a = np.zeros(math.prod(shape), dtype=np.float64)
            a[z[k + ".idx"].astype(np.int64)] = z[k + ".val"]
            return a.reshape(shape)

        emb = [dense("emb." + f) for f in cfg["fields"]]
        W_out = dense("out.W")
        b_out = dense("out.b")
        layers = []
        for l, heads in enumerate(cfg["layers"]):
            lay: dict[str, Any] = {
                "heads": heads,
                "W1": dense(f"L{l}.W1"),
                "b1": dense(f"L{l}.b1"),
                "W2": dense(f"L{l}.W2"),
                "b2": dense(f"L{l}.b2"),
            }
            if heads:
                for k in ("Wq", "Wk", "Wv", "Wo"):
                    lay[k] = dense(f"L{l}.{k}")
            layers.append(lay)
    return {"cfg": cfg, "emb": emb, "W_out": W_out, "b_out": b_out, "layers": layers}


def _tokenize_prompt(name: str, cfg: dict[str, Any], arg: Any) -> list[list[int]]:
    io = cfg["io"]
    if name == "maze":
        rows = str(arg).split("\n")
        return [[r, c, io["charset"].index(ch)] for r, row in enumerate(rows) for c, ch in enumerate(row)]
    if name == "hanoi":
        n = int(arg)
        ix = {s: i for i, s in enumerate(io["vocab"]) if s}
        words = [f"INIT({d},0)" for d in range(n, 0, -1)] + [f"GOAL({d},2)" for d in range(n, 0, -1)]
        return [[ix[w]] for w in [f"N={n}"] + words + ["SEP"]]
    if name == "graph":
        body, q = str(arg).split("?")
        k, words, (a, b) = io["kind"], body.split(), map(int, q.split())
        listed = {int(w) for w in words if ">" not in w}
        ids = sorted({int(x) for w in words for x in w.split(">")} | {a, b})
        toks = [[k["node"], i, i] for i in ids if i not in listed]
        toks += [
            [k["edge"]] + [int(x) for x in w.split(">")] if ">" in w else [k["node"], int(w), int(w)]
            for w in words
        ]
        return toks + [[k["query"], a, b]]
    raise ValueError(f"Unsupported task: {name}")


def _loops(rule: dict[str, Any], n_input: int, n_call: int) -> int:
    n = n_call if rule["n"] == "call" else n_input
    return int(rule["const"] + rule["lin"] * -(-n // rule["div"]) + rule["log2"] * math.ceil(math.log2(max(n, 2))))


def _rmsnorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    denom = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / denom


def _softmax_safe(s: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        s = s - np.max(s, axis=-1, keepdims=True)
        e = np.exp(s)
        return e / np.sum(e, axis=-1, keepdims=True)


def _embed_tokens_np(net: dict[str, Any], toks: list[list[int]], p0: int) -> np.ndarray:
    T = np.asarray(toks, dtype=np.int64)
    if net["cfg"].get("position_field"):
        T = np.column_stack([T, p0 + np.arange(len(T))])
    h = net["emb"][0][T[:, 0]].copy()
    for f in range(1, T.shape[1]):
        h = h + net["emb"][f][T[:, f]]
    return h


def _prepare_weights_np(net: dict[str, Any], w_mode: str) -> dict[str, Any]:
    if w_mode == "fp64":
        return net

    def qw(a: np.ndarray) -> np.ndarray:
        return _quantize(a, w_mode)

    return {
        "cfg": net["cfg"],
        "emb": [qw(e) for e in net["emb"]],
        "W_out": qw(net["W_out"]),
        "b_out": qw(net["b_out"]),
        "layers": [
            {k: (qw(v) if isinstance(v, np.ndarray) else v) for k, v in L.items()}
            for L in net["layers"]
        ],
    }


def _forward_perturbed_np(
    pnet: dict[str, Any],
    toks: list[list[int]],
    p0: int,
    R: int,
    *,
    act_mode: str = "fp64",
    caches=None,
    init_h_override: np.ndarray | None = None,
    noise_sigma: float = 0.0,
    noise_mode: str = "none",
    rng: np.random.Generator | None = None,
    rmsnorm_mode: str = "none",
) -> np.ndarray:
    c = pnet["cfg"]
    compute_dt = np.float32 if act_mode == "fp32" else np.float64

    def qa(x: np.ndarray) -> np.ndarray:
        if act_mode in ("fp64", "fp32"):
            return x.astype(compute_dt, copy=False)
        return _quantize(x, act_mode)

    if init_h_override is not None:
        h = qa(init_h_override.copy())
    else:
        T = np.asarray(toks, dtype=np.int64)
        if c.get("position_field"):
            T = np.column_stack([T, p0 + np.arange(len(T))])
        h = qa(pnet["emb"][0][T[:, 0]].copy())
        for f in range(1, T.shape[1]):
            h = qa(h + pnet["emb"][f][T[:, f]])
        if noise_sigma > 0.0 and noise_mode == "init_only":
            assert rng is not None
            h = qa(h + rng.standard_normal(h.shape) * noise_sigma)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for t in range(R):
            for l, L in enumerate(pnet["layers"]):
                h_in = qa(_rmsnorm(h)) if rmsnorm_mode == "pre_block" else h
                if L["heads"]:
                    W_q = L["Wq"].astype(compute_dt, copy=False)
                    W_k = L["Wk"].astype(compute_dt, copy=False)
                    W_v = L["Wv"].astype(compute_dt, copy=False)
                    W_o = L["Wo"].astype(compute_dt, copy=False)
                    Q = qa(h_in @ W_q)
                    K = qa(h_in @ W_k)
                    V = qa(h_in @ W_v)
                    if caches is not None:
                        cache = caches[t][l]
                        n_prev = cache["n"]
                        n_new = n_prev + len(K)
                        if cache["K"] is None or n_new > len(cache["K"]):
                            K2 = np.empty((2 * n_new, K.shape[1]), dtype=compute_dt)
                            V2 = np.empty((2 * n_new, V.shape[1]), dtype=compute_dt)
                            if n_prev:
                                K2[:n_prev], V2[:n_prev] = cache["K"][:n_prev], cache["V"][:n_prev]
                            cache["K"], cache["V"] = K2, V2
                        cache["K"][n_prev:n_new], cache["V"][n_prev:n_new] = K, V
                        cache["n"] = n_new
                        K, V = cache["K"][:n_new], cache["V"][:n_new]

                    outs, q0, v0 = [], 0, 0
                    for dk, dv in L["heads"]:
                        scale = compute_dt(1.0 / math.sqrt(dk))
                        s = qa((Q[:, q0:q0 + dk] @ K[:, q0:q0 + dk].T) * scale)
                        attn_w = qa(_softmax_safe(s.astype(np.float32 if act_mode != "fp64" else np.float64)))
                        outs.append(qa(attn_w.astype(compute_dt) @ V[:, v0:v0 + dv]))
                        q0, v0 = q0 + dk, v0 + dv
                    h = qa(h + qa(np.hstack(outs) @ W_o))
                    if rmsnorm_mode == "post_block":
                        h = qa(_rmsnorm(h))

                h_mlp_in = qa(_rmsnorm(h)) if rmsnorm_mode == "pre_block" else h
                W1 = L["W1"].astype(compute_dt, copy=False)
                b1 = L["b1"].astype(compute_dt, copy=False)
                W2 = L["W2"].astype(compute_dt, copy=False)
                b2 = L["b2"].astype(compute_dt, copy=False)
                hid = np.maximum(qa(h_mlp_in @ W1 + b1), 0.0)
                mlp_out = qa(hid @ W2 + b2)
                h = qa(h + mlp_out)
                if rmsnorm_mode == "post_block":
                    h = qa(_rmsnorm(h))

            if noise_sigma > 0.0 and noise_mode == "per_loop":
                assert rng is not None
                h = qa(h + rng.standard_normal(h.shape) * noise_sigma)

        rows = h if c["readout"] == "all" else h[-1:]
        logits = rows @ pnet["W_out"].astype(compute_dt, copy=False) + pnet["b_out"].astype(compute_dt, copy=False)
        return np.asarray(logits, dtype=np.float64)


def _eval_task_instances(
    task: str,
    pnet: dict[str, Any],
    instances: list[Any],
    *,
    act_mode: str = "fp64",
    loop_mult: int = 1,
    noise_sigma: float = 0.0,
    noise_mode: str = "none",
    rmsnorm_mode: str = "none",
    seed: int = 0,
    init_overrides: list[np.ndarray] | None = None,
) -> tuple[float, list[str]]:
    cfg = pnet["cfg"]
    io = cfg["io"]
    rng = np.random.default_rng(seed)
    n_ok = 0
    outputs: list[str] = []

    for idx, arg in enumerate(instances):
        toks = _tokenize_prompt(task, cfg, arg)
        R = _loops(cfg["loops"], len(toks), len(toks)) * loop_mult
        ov = init_overrides[idx] if init_overrides is not None else None

        if cfg["readout"] == "all":
            lg = _forward_perturbed_np(
                pnet, toks, 0, R, act_mode=act_mode, init_h_override=ov,
                noise_sigma=noise_sigma, noise_mode=noise_mode, rng=rng, rmsnorm_mode=rmsnorm_mode,
            )
            if not np.all(np.isfinite(lg)):
                sol = ""
                ok = False
            else:
                gen = [int(i) for i in lg.argmax(-1)]
                n = math.isqrt(len(gen))
                sol = "\n".join("".join(io["charset"][i] for i in gen[r * n:(r + 1) * n]) for r in range(n))
                ok = verify_maze_solution(str(arg), sol)
            outputs.append(sol)
            n_ok += int(ok)
        else:
            caches = [[{"n": 0, "K": None, "V": None} for _ in pnet["layers"]] for _ in range(R)] if cfg["mask"] != "none" else None
            if cfg["mask"] == "causal":
                lg = None
                for i, tk in enumerate(toks):
                    tk_ov = ov[i:i + 1] if ov is not None else None
                    lg = _forward_perturbed_np(
                        pnet, [tk], i, R, act_mode=act_mode, caches=caches, init_h_override=tk_ov,
                        noise_sigma=noise_sigma, noise_mode=noise_mode, rng=rng, rmsnorm_mode=rmsnorm_mode,
                    )
            else:
                lg = _forward_perturbed_np(
                    pnet, toks, 0, R, act_mode=act_mode, caches=caches, init_h_override=ov,
                    noise_sigma=noise_sigma, noise_mode=noise_mode, rng=rng, rmsnorm_mode=rmsnorm_mode,
                )
            gen: list[int] = []
            seq = list(toks)
            bad = not np.all(np.isfinite(lg))
            # Cap corrupted non-halting generations (hanoi.json sets max_tokens=2097152, max_new=None)
            cap_new = cfg.get("max_new")
            if cap_new is None:
                cap_new = (2 ** int(arg) + 16) if task == "hanoi" else 16
            while not bad:
                o = int(np.argmax(lg[-1]))
                if o == cfg["end_id"]:
                    break
                gen.append(o)
                if len(gen) >= cap_new or len(seq) >= cfg["max_tokens"]:
                    break
                tok = list(cfg["out_token"]["template"])
                tok[cfg["out_token"]["slot"]] = o
                seq.append(tok)
                if caches is not None:
                    lg = _forward_perturbed_np(
                        pnet, [tok], len(seq) - 1, R, act_mode=act_mode, caches=caches,
                        noise_sigma=noise_sigma, noise_mode=noise_mode, rng=rng, rmsnorm_mode=rmsnorm_mode,
                    )
                else:
                    lg = _forward_perturbed_np(
                        pnet, seq, 0, _loops(cfg["loops"], len(toks), len(seq)) * loop_mult,
                        act_mode=act_mode, noise_sigma=noise_sigma, noise_mode=noise_mode,
                        rng=rng, rmsnorm_mode=rmsnorm_mode,
                    )
                if not np.all(np.isfinite(lg)):
                    bad = True
                    break

            if bad or not gen:
                sol = ""
                ok = False
            elif task == "hanoi":
                sol = "moves = [" + ", ".join(io["vocab"][i] for i in gen) + "]"
                ok = verify_hanoi_moves(int(arg), sol)
            elif task == "graph":
                sol = "unreachable" if gen[0] == io.get("unreachable") else str(gen[0])
                ok = (sol == _graph_truth(toks, io))
            else:
                sol = str(gen)
                ok = False
            outputs.append(sol)
            n_ok += int(ok)

    return n_ok / max(1, len(instances)), outputs


def _eval_condition(
    nets: dict[str, dict[str, Any]],
    datasets: dict[str, list[Any]],
    *,
    w_mode: str = "fp64",
    act_mode: str = "fp64",
    noise_sigma: float = 0.0,
    noise_mode: str = "none",
    rmsnorm_mode: str = "none",
    check_fp: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in ("maze", "hanoi", "graph"):
        pnet = _prepare_weights_np(nets[task], w_mode)
        acc_r, sol_r = _eval_task_instances(
            task, pnet, datasets[task], act_mode=act_mode, loop_mult=1,
            noise_sigma=noise_sigma, noise_mode=noise_mode, rmsnorm_mode=rmsnorm_mode, seed=42,
        )
        fp_surv = None
        if check_fp:
            fp_inst = datasets[task][:3] if task == "hanoi" else datasets[task]
            _, sol_2r = _eval_task_instances(
                task, pnet, fp_inst, act_mode=act_mode, loop_mult=2,
                noise_sigma=noise_sigma, noise_mode=noise_mode, rmsnorm_mode=rmsnorm_mode, seed=42,
            )
            fp_surv = round(sum(a == b for a, b in zip(sol_r[:len(fp_inst)], sol_2r)) / len(fp_inst), 4)
        out[task] = {"accuracy": round(acc_r, 4), "fixed_point_2R_survival": fp_surv}
    out["mean_accuracy"] = round(sum(out[t]["accuracy"] for t in ("maze", "hanoi", "graph")) / 3.0, 4)
    return out


# ------------------------------------------------------------------ Experiment 4: D=1024 Linear Interface

def _run_linear_interface_experiment(
    nets: dict[str, dict[str, Any]],
    cal_data: dict[str, list[Any]],
    test_data: dict[str, list[Any]],
    D: int = 1024,
    etas: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
) -> list[dict[str, Any]]:
    rng_np = np.random.default_rng(1234)
    rows: list[dict[str, Any]] = []

    projections: dict[str, np.ndarray] = {}
    h0_cal: dict[str, np.ndarray] = {}
    h0_test_list: dict[str, list[np.ndarray]] = {}
    for task in ("maze", "hanoi", "graph"):
        net = nets[task]
        d = int(net["cfg"]["d_model"])
        q, _ = np.linalg.qr(rng_np.standard_normal((D, d)))
        projections[task] = q[:, :d]
        cal_chunks = [_embed_tokens_np(net, _tokenize_prompt(task, net["cfg"], arg), 0) for arg in cal_data[task]]
        h0_cal[task] = np.vstack(cal_chunks)
        h0_test_list[task] = [
            _embed_tokens_np(net, _tokenize_prompt(task, net["cfg"], arg), 0)
            for arg in test_data[task]
        ]

    for eta in etas:
        rec: dict[str, Any] = {"eta": float(eta), "D": D}
        accs, snrs, rmses = [], [], []
        for task in ("maze", "hanoi", "graph"):
            U = projections[task]
            X0_cal = h0_cal[task]
            reps = max(1, (2 * D) // X0_cal.shape[0] + 1)
            X_rep = np.tile(X0_cal, (reps, 1))
            Z_cal = rng_np.standard_normal((X_rep.shape[0], D))
            H_cal = X_rep @ U.T + eta * Z_cal
            lam = max(1e-6, 0.01 * (eta ** 2) * X_rep.shape[0])
            W_in = np.linalg.solve(H_cal.T @ H_cal + lam * np.eye(D), H_cal.T @ X_rep)

            ov_list: list[np.ndarray] = []
            sq_err = sq_sig = 0.0
            n_el = 0
            for X0_t in h0_test_list[task]:
                Z_t = rng_np.standard_normal((X0_t.shape[0], D))
                H_t = X0_t @ U.T + eta * Z_t
                X_hat = H_t @ W_in
                err = X_hat - X0_t
                sq_err += float(np.sum(err * err))
                sq_sig += float(np.sum(X0_t * X0_t))
                n_el += X0_t.size
                ov_list.append(X_hat)

            rmse = math.sqrt(sq_err / max(1, n_el))
            snr_db = 10.0 * math.log10(sq_sig / max(sq_err, 1e-18)) if sq_err > 1e-18 else 180.0
            acc, _ = _eval_task_instances(
                task, nets[task], test_data[task], act_mode="fp64", loop_mult=1, init_overrides=ov_list
            )
            rec[task] = {"accuracy": round(acc, 4), "rmse": round(rmse, 5), "snr_db": round(snr_db, 2)}
            accs.append(acc)
            rmses.append(rmse)
            snrs.append(snr_db)
        rec["mean_accuracy"] = round(sum(accs) / 3.0, 4)
        rec["mean_rmse"] = round(sum(rmses) / 3.0, 5)
        rec["mean_snr_db"] = round(sum(snrs) / 3.0, 2)
        rows.append(rec)
    return rows


def run_stress_suite(repo_dir: Path, out_path: Path) -> dict[str, Any]:
    _verify_bf16_matches_torch()
    net_dir = repo_dir / "weights" / "networks"
    nets = {name: _load_network_np(net_dir, name) for name in ("maze", "hanoi", "graph")}

    test_data = {
        "maze": [_make_seeded_maze(9, 100 + i) for i in range(10)],
        "hanoi": [3, 4, 5, 6],
        "graph": [_make_seeded_graph(10, 200 + i) for i in range(15)],
    }
    cal_data = {
        "maze": [_make_seeded_maze(9, 500 + i) for i in range(8)],
        "hanoi": [3, 4],
        "graph": [_make_seeded_graph(10, 600 + i) for i in range(10)],
    }

    t0 = time.perf_counter()

    print("[1/4] Precision cliff...", flush=True)
    precision_specs = [
        ("fp64", "fp64", "fp64"),
        ("fp32", "fp32", "fp32"),
        ("fp16", "fp16", "fp16"),
        ("bf16_weights_only", "bf16", "fp64"),
        ("bf16_acts_only", "fp64", "bf16"),
        ("bf16_both", "bf16", "bf16"),
    ]
    precision_results = {}
    for label, wm, am in precision_specs:
        precision_results[label] = _eval_condition(nets, test_data, w_mode=wm, act_mode=am, check_fp=True)

    if precision_results["fp64"]["mean_accuracy"] < 1.0:
        raise RuntimeError(f"Baseline fp64 did not achieve 100%: {precision_results['fp64']}")

    print("[2/4] Gaussian noise cliff...", flush=True)
    sigmas = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    noise_results: dict[str, list[dict[str, Any]]] = {"init_only": [], "per_loop": []}
    for mode in ("init_only", "per_loop"):
        for sig in sigmas:
            res = _eval_condition(
                nets, test_data, w_mode="fp64", act_mode="fp64",
                noise_sigma=sig, noise_mode=mode, check_fp=(sig in (0.0, 1e-3, 1e-2, 1e-1)),
            )
            noise_results[mode].append({"sigma": sig, **res})

    print("[3/4] RMSNorm injection...", flush=True)
    rmsnorm_results = {}
    for mode in ("none", "pre_block", "post_block"):
        rmsnorm_results[mode] = _eval_condition(
            nets, test_data, w_mode="fp64", act_mode="fp64", rmsnorm_mode=mode, check_fp=True,
        )

    print("[4/4] D=1024 linear projection interface...", flush=True)
    linear_iface = _run_linear_interface_experiment(nets, cal_data, test_data, D=1024)

    elapsed_s = round(time.perf_counter() - t0, 2)
    payload = {
        "repo_dir": str(repo_dir),
        "elapsed_s": elapsed_s,
        "dataset_sizes": {"maze_9x9": 10, "hanoi_3_to_6": 4, "graph_10node": 15},
        "maze_size_1k_note": (
            "maze_size_1k (928 params, 136 NNZ) hardcodes 30x30 newline stride & R=1415 loops "
            "(~70s/instance on CPU); stress sweep uses size-agnostic maze (19,936 params), "
            "hanoi (427,654 total / 3,686 NNZ), and graph (6,966 params)."
        ),
        "precision_cliff": precision_results,
        "gaussian_noise_cliff": noise_results,
        "rmsnorm_injection": rmsnorm_results,
        "linear_projection_interface_D1024": linear_iface,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)

    print("\n=== 1. Precision Cliff ===", flush=True)
    print("| Regime | Maze | Hanoi | Graph | Mean | Maze 2R-FP |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for label, r in precision_results.items():
        print(
            f"| `{label}` | {r['maze']['accuracy']:.1%} | {r['hanoi']['accuracy']:.1%} | "
            f"{r['graph']['accuracy']:.1%} | {r['mean_accuracy']:.1%} | {r['maze']['fixed_point_2R_survival']:.1%} |"
        )
    print("\n=== 2. Gaussian Noise Cliff ===")
    for mode in ("init_only", "per_loop"):
        print(f"-- mode={mode} --")
        for row in noise_results[mode]:
            print(
                f"  sigma={row['sigma']:<7g} | maze={row['maze']['accuracy']:.1%} "
                f"hanoi={row['hanoi']['accuracy']:.1%} graph={row['graph']['accuracy']:.1%} "
                f"mean={row['mean_accuracy']:.1%}"
            )
    print("\n=== 3. RMSNorm Injection ===")
    for mode, r in rmsnorm_results.items():
        print(
            f"  rmsnorm={mode:<10s} | maze={r['maze']['accuracy']:.1%} "
            f"hanoi={r['hanoi']['accuracy']:.1%} graph={r['graph']['accuracy']:.1%} "
            f"mean={r['mean_accuracy']:.1%}"
        )
    print("\n=== 4. Linear Interface D=1024 (Qwen3.5-0.8B width) ===")
    for row in linear_iface:
        print(
            f"  eta={row['eta']:<5g} | rmse={row['mean_rmse']:.4f} | snr={row['mean_snr_db']:5.1f} dB | "
            f"maze={row['maze']['accuracy']:.1%} hanoi={row['hanoi']['accuracy']:.1%} graph={row['graph']['accuracy']:.1%}"
        )
    print(f"\nSaved stress results to {out_path} in {elapsed_s} s", flush=True)
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-dir", default=str(ROOT / "work" / "LoopedWSP"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "looped_wsp_coproc" / "results_stress_cpu.json"))
    a = ap.parse_args(argv)
    run_stress_suite(Path(a.repo_dir), Path(a.out))


if __name__ == "__main__":
    main()
