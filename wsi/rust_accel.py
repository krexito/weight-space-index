"""Python `ctypes` bridge to the native multithreaded SIMD Rust engine (`wsi_core`).

Automatically loads `rust/target/release/wsi_core.dll` (Windows), `libwsi_core.so` (Linux),
or `libwsi_core.dylib` (macOS) if compiled, providing zero-copy `BF16` / `F16` / `F32`
residual matrix delta analysis across all CPU cores without intermediate tensor allocations.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parent.parent
_RELEASE_DIR = _ROOT / "rust" / "target" / "release"


class _ResidualDeltaStats(ctypes.Structure):
    _fields_ = [
        ("old_norm", ctypes.c_double),
        ("new_norm", ctypes.c_double),
        ("delta_norm", ctypes.c_double),
        ("overall_cosine_new_old", ctypes.c_double),
        ("values_changed", ctypes.c_uint64),
        ("changed_units_count", ctypes.c_uint64),
        ("top_modified_unit", ctypes.c_uint64),
        ("s0", ctypes.c_double),
        ("old_energy", ctypes.c_double),
        ("new_energy", ctypes.c_double),
    ]


def _load_rust_lib() -> ctypes.CDLL | None:
    if sys.platform == "win32":
        cand = _RELEASE_DIR / "wsi_core.dll"
    elif sys.platform == "darwin":
        cand = _RELEASE_DIR / "libwsi_core.dylib"
    else:
        cand = _RELEASE_DIR / "libwsi_core.so"

    if not cand.exists():
        return None
    try:
        lib = ctypes.CDLL(str(cand))
        lib.wsi_analyze_residual_delta.argtypes = [
            ctypes.c_void_p,  # old_ptr
            ctypes.c_int32,   # old_dtype
            ctypes.c_void_p,  # new_ptr
            ctypes.c_int32,   # new_dtype
            ctypes.c_size_t,  # hidden
            ctypes.c_size_t,  # n_units
            ctypes.c_int32,   # out_axis
            ctypes.POINTER(_ResidualDeltaStats),
            ctypes.c_void_p,  # out_top_u
            ctypes.c_void_p,  # out_old_col
            ctypes.c_void_p,  # out_new_col
        ]
        lib.wsi_analyze_residual_delta.restype = ctypes.c_int32
        return lib
    except Exception:
        return None


_LIB = _load_rust_lib()

_DTYPE_MAP = {
    torch.float32: 0,
    torch.bfloat16: 1,
    torch.float16: 2,
}


def has_rust_core() -> bool:
    return _LIB is not None


def rust_analyze_residual_delta(
    T_old: torch.Tensor,
    T_new: torch.Tensor,
    out_axis: int,
) -> dict[str, Any] | None:
    """Run zero-copy multithreaded Rust analysis on 2D `T_old` and `T_new` (`F32`, `BF16`, or `F16`)."""
    if _LIB is None or T_old.dim() != 2 or T_new.dim() != 2:
        return None
    old_dt = _DTYPE_MAP.get(T_old.dtype)
    new_dt = _DTYPE_MAP.get(T_new.dtype)
    if old_dt is None or new_dt is None:
        return None

    T_old_c = T_old.contiguous()
    T_new_c = T_new.contiguous()
    if out_axis == 0:
        hidden, n_units = T_old_c.shape
    else:
        n_units, hidden = T_old_c.shape

    stats = _ResidualDeltaStats()
    top_u = torch.empty(hidden, dtype=torch.float32)
    old_col = torch.empty(hidden, dtype=torch.float32)
    new_col = torch.empty(hidden, dtype=torch.float32)

    rc = _LIB.wsi_analyze_residual_delta(
        ctypes.c_void_p(T_old_c.data_ptr()),
        ctypes.c_int32(old_dt),
        ctypes.c_void_p(T_new_c.data_ptr()),
        ctypes.c_int32(new_dt),
        ctypes.c_size_t(hidden),
        ctypes.c_size_t(n_units),
        ctypes.c_int32(out_axis),
        ctypes.byref(stats),
        ctypes.c_void_p(top_u.data_ptr()),
        ctypes.c_void_p(old_col.data_ptr()),
        ctypes.c_void_p(new_col.data_ptr()),
    )
    if rc != 0:
        return None

    return {
        "old_norm": float(stats.old_norm),
        "new_norm": float(stats.new_norm),
        "delta_norm": float(stats.delta_norm),
        "overall_cosine_new_old": float(stats.overall_cosine_new_old),
        "values_changed": int(stats.values_changed),
        "changed_units_count": int(stats.changed_units_count),
        "total_units_count": int(n_units),
        "top_modified_unit": int(stats.top_modified_unit),
        "s0": float(stats.s0),
        "old_energy": float(stats.old_energy),
        "new_energy": float(stats.new_energy),
        "top_u": top_u,
        "old_col": old_col,
        "new_col": new_col,
    }
