"""Early loop stop for vLLM's V2 model runner (vLLM >= 0.3x).

moi_loop_router_vllm.LoopKillLogitsProcessor uses the old AdapterLogitsProcessor API. That class no longer exists in
vLLM 0.34, so the import failed silently and every response that looped ran on to the token cap: in the blind
discovery run (#13) `early_loop_stop=False` was printed for every config, and most of the 9 hours went into
generating dead loops to 16k tokens.

This processor implements the same rule for the new API (`vllm.v1.worker.gpu.sample.logits_processor`). It forces
EOS once >= 90% of the last 768 response tokens are repeated 16-grams (moi_loop_router_vllm.LoopKillState). It reads
the token history in batches every STRIDE decode steps, which costs one device sync per STRIDE steps, so the stop can
come up to STRIDE tokens after the rule first holds. Grading cuts every response at the first point where the rule
holds (moi_loop_router_vllm.grade -> analyse -> first_window_hit), so the graded result is the same as with no early
stop.

Enable per request with SamplingParams(extra_args={"loop_kill_eos": <eos id>}).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moi_loop_router_vllm as lr  # noqa: E402

STRIDE = 32

try:  # vLLM V2 model runner interface (vLLM 0.34)
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor as _Base
except Exception:  # noqa: BLE001  (no vLLM locally, or an older one)
    _Base = None


def _sync_ok():
    try:
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed

        return gpu_sync_allowed()
    except Exception:  # noqa: BLE001
        import contextlib

        return contextlib.nullcontext()


class _LoopKillCore:
    """Per-slot host state; the vLLM-free part, tested locally with a fake request state."""

    def __init__(self, req_states, stride: int = STRIDE):
        self.rs = req_states
        n = req_states.max_num_reqs
        self.on = np.zeros(n, dtype=bool)
        self.hit = np.zeros(n, dtype=bool)
        self.eos = np.zeros(n, dtype=np.int64)
        self.state: dict[int, lr.LoopKillState] = {}
        self.toks: dict[int, list[int]] = {}
        self.stride = stride
        self.step = 0

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        ea = getattr(sampling_params, "extra_args", None) or {}
        if "loop_kill_eos" in ea and not isinstance(ea["loop_kill_eos"], int):
            raise ValueError("loop_kill_eos must be an int token id")

    def add_request(self, req_idx: int, sampling_params) -> bool:
        ea = getattr(sampling_params, "extra_args", None) or {}
        on = "loop_kill_eos" in ea
        self.on[req_idx] = on
        self.hit[req_idx] = False
        self.state.pop(req_idx, None)
        self.toks.pop(req_idx, None)
        if on:
            self.eos[req_idx] = int(ea["loop_kill_eos"])
            self.state[req_idx] = lr.LoopKillState(int(ea["loop_kill_eos"]))
            self.toks[req_idx] = []
        return on

    def _read_new(self, slots: np.ndarray, cur: np.ndarray) -> list[list[int]]:
        """New committed response tokens of each slot since the last read (the newest token is left for later)."""
        p = self.rs.prompt_len.np[slots].astype(np.int64)
        start = p + np.array([len(self.toks[int(s)]) for s in slots], dtype=np.int64)
        end = cur.astype(np.int64) - 1
        n = np.maximum(end - start, 0)
        if n.max(initial=0) == 0:
            return [[] for _ in slots]
        import torch

        all_tok = self.rs.all_token_ids.gpu
        dev = all_tok.device
        width = int(n.max())
        idx = torch.as_tensor(start, device=dev)[:, None] + torch.arange(width, device=dev)[None, :]
        idx = idx.clamp(max=all_tok.shape[1] - 1)
        with _sync_ok():
            block = all_tok[torch.as_tensor(slots, device=dev)[:, None], idx].cpu().numpy()
        return [block[i, : n[i]].tolist() for i in range(len(slots))]

    def update(self, slots: np.ndarray, cur: np.ndarray) -> None:
        keep = ~self.hit[slots]
        live, cur_live = slots[keep], cur[keep]
        if live.size == 0:
            return
        for s, new in zip(live, self._read_new(live, cur_live)):
            s = int(s)
            toks, st = self.toks[s], self.state[s]
            for t in new:
                toks.append(int(t))
                if st.killed(toks):
                    self.hit[s] = True
                    break

    def apply(self, logits, ctx):
        slots = np.asarray(ctx.idx_mapping_np)
        rows = np.flatnonzero(self.on[slots])
        if rows.size == 0 or logits.shape[0] != slots.shape[0]:  # nothing to do, or speculative decoding rows
            return logits
        self.step += 1
        if self.step % self.stride == 0:
            self.update(slots[rows], np.asarray(ctx.seq_lens_upper_bound_np)[rows])
        dead = rows[self.hit[slots[rows]]]
        if dead.size:
            import torch

            r = torch.as_tensor(dead, device=logits.device)
            e = torch.as_tensor(self.eos[slots[dead]], device=logits.device)
            logits[r] = float("-inf")
            logits[r, e] = 0.0
        return logits


if _Base is not None:
    class LoopKillV2(_Base):
        """vLLM V2 logits processor: the kill rule of moi_loop_router_vllm, checked every STRIDE steps."""

        def __init__(self, vllm_config, req_states):
            self.core = _LoopKillCore(req_states)

        @classmethod
        def validate_params(cls, sampling_params) -> None:
            _LoopKillCore.validate_params(sampling_params)

        def add_request(self, req_idx, sampling_params) -> bool:
            return self.core.add_request(req_idx, sampling_params)

        def apply(self, logits, ctx):
            return self.core.apply(logits, ctx)
else:
    LoopKillV2 = None


def processor():
    """The early-stop class this vLLM accepts: the V2 one, else the old adapter one, else None."""
    return LoopKillV2 if LoopKillV2 is not None else lr.LoopKillLogitsProcessor
