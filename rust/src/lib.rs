//! Zero-copy, multithreaded SIMD Rust engine for Weight-Space Index (`wsi`).
//!
//! Provides C-ABI entrypoints (`wsi_analyze_residual_delta`, `wsi_matvec_scores`) and native Rust
//! functions that operate directly on memory-mapped `.safetensors` buffers (`F32`, `BF16`, `F16`)
//! across all CPU cores without intermediate tensor allocations or `BF16 -> FP32` buffer copies.

use std::slice;
use std::thread;

/// Dtype codes passed across the C-ABI boundary:
/// 0 = Float32 (4 bytes)
/// 1 = BFloat16 (2 bytes)
/// 2 = Float16 (2 bytes)
#[inline(always)]
fn read_elem(ptr: *const u8, dtype: i32, idx: usize) -> f32 {
    unsafe {
        match dtype {
            0 => *(ptr as *const f32).add(idx),
            1 => {
                let bits = *(ptr as *const u16).add(idx);
                f32::from_bits((bits as u32) << 16)
            }
            2 => {
                let h = *(ptr as *const u16).add(idx);
                f16_to_f32(h)
            }
            _ => 0.0,
        }
    }
}

#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            f32::from_bits(sign | ((e as u32) << 23) | (m << 13))
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

#[repr(C)]
pub struct ResidualDeltaStats {
    pub old_norm: f64,
    pub new_norm: f64,
    pub delta_norm: f64,
    pub overall_cosine_new_old: f64,
    pub values_changed: u64,
    pub changed_units_count: u64,
    pub top_modified_unit: u64,
    pub s0: f64,
    pub old_energy: f64,
    pub new_energy: f64,
}

/// Analyze the difference `dW = W_new - W_old` between two 2D residual matrices in a single parallel pass
/// + fast active-subspace power SVD, writing:
/// - scalar statistics into `out_stats`
/// - principal left singular vector `u0 * s0` into `out_top_u` (`[hidden]`)
/// - if `0 < changed_units_count <= 64`, the top modified unit's old & new column vectors into
///   `out_old_col` (`[hidden]`) and `out_new_col` (`[hidden]`).
#[no_mangle]
pub extern "C" fn wsi_analyze_residual_delta(
    old_ptr: *const u8,
    old_dtype: i32,
    new_ptr: *const u8,
    new_dtype: i32,
    hidden: usize,
    n_units: usize,
    out_axis: i32,
    out_stats: *mut ResidualDeltaStats,
    out_top_u: *mut f32,
    out_old_col: *mut f32,
    out_new_col: *mut f32,
) -> i32 {
    if old_ptr.is_null() || new_ptr.is_null() || out_stats.is_null() || out_top_u.is_null() {
        return -1;
    }

    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(16);

    let old_addr = old_ptr as usize;
    let new_addr = new_ptr as usize;

    let mut old_norm_sq = 0.0f64;
    let mut new_norm_sq = 0.0f64;
    let mut delta_norm_sq = 0.0f64;
    let mut dot_new_old = 0.0f64;
    let mut values_changed = 0u64;
    let mut col_delta_sq = vec![0.0f64; n_units];

    if out_axis == 1 {
        // Shape on disk is [n_units, hidden] (e.g. GPT-2 Conv1D): each row `u` is a contiguous column of length `hidden`
        let chunk_u = (n_units + n_threads - 1) / n_threads;
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for t in 0..n_threads {
                let u_start = t * chunk_u;
                let u_end = (u_start + chunk_u).min(n_units);
                if u_start >= u_end {
                    continue;
                }
                handles.push(s.spawn(move || {
                    let op = old_addr as *const u8;
                    let np = new_addr as *const u8;
                    let mut loc_old_sq = 0.0f64;
                    let mut loc_new_sq = 0.0f64;
                    let mut loc_d_sq = 0.0f64;
                    let mut loc_dot = 0.0f64;
                    let mut loc_changed = 0u64;
                    let mut loc_col_sq = vec![0.0f64; u_end - u_start];

                    for u in u_start..u_end {
                        let base = u * hidden;
                        let mut c_sq = 0.0f64;
                        for h in 0..hidden {
                            let idx = base + h;
                            let vo = read_elem(op, old_dtype, idx) as f64;
                            let vn = read_elem(np, new_dtype, idx) as f64;
                            let dv = vn - vo;
                            loc_old_sq += vo * vo;
                            loc_new_sq += vn * vn;
                            loc_dot += vn * vo;
                            if dv != 0.0 {
                                loc_changed += 1;
                                c_sq += dv * dv;
                            }
                        }
                        loc_d_sq += c_sq;
                        loc_col_sq[u - u_start] = c_sq;
                    }
                    (u_start, loc_old_sq, loc_new_sq, loc_d_sq, loc_dot, loc_changed, loc_col_sq)
                }));
            }
            for h in handles {
                let (u_start, o_sq, n_sq, d_sq, dot, chg, c_sq) = h.join().unwrap();
                old_norm_sq += o_sq;
                new_norm_sq += n_sq;
                delta_norm_sq += d_sq;
                dot_new_old += dot;
                values_changed += chg;
                col_delta_sq[u_start..u_start + c_sq.len()].copy_from_slice(&c_sq);
            }
        });
    } else {
        // Shape on disk is [hidden, n_units] (e.g. Llama, Qwen, LFM3): partition across rows `h` for stride-1 access
        let chunk_h = (hidden + n_threads - 1) / n_threads;
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_threads);
            for t in 0..n_threads {
                let h_start = t * chunk_h;
                let h_end = (h_start + chunk_h).min(hidden);
                if h_start >= h_end {
                    continue;
                }
                handles.push(s.spawn(move || {
                    let op = old_addr as *const u8;
                    let np = new_addr as *const u8;
                    let mut loc_old_sq = 0.0f64;
                    let mut loc_new_sq = 0.0f64;
                    let mut loc_d_sq = 0.0f64;
                    let mut loc_dot = 0.0f64;
                    let mut loc_changed = 0u64;
                    let mut loc_col_sq = vec![0.0f64; n_units];

                    for h in h_start..h_end {
                        let base = h * n_units;
                        for u in 0..n_units {
                            let idx = base + u;
                            let vo = read_elem(op, old_dtype, idx) as f64;
                            let vn = read_elem(np, new_dtype, idx) as f64;
                            let dv = vn - vo;
                            loc_old_sq += vo * vo;
                            loc_new_sq += vn * vn;
                            loc_dot += vn * vo;
                            if dv != 0.0 {
                                loc_changed += 1;
                                let d2 = dv * dv;
                                loc_d_sq += d2;
                                loc_col_sq[u] += d2;
                            }
                        }
                    }
                    (loc_old_sq, loc_new_sq, loc_d_sq, loc_dot, loc_changed, loc_col_sq)
                }));
            }
            for h in handles {
                let (o_sq, n_sq, d_sq, dot, chg, c_sq) = h.join().unwrap();
                old_norm_sq += o_sq;
                new_norm_sq += n_sq;
                delta_norm_sq += d_sq;
                dot_new_old += dot;
                values_changed += chg;
                for u in 0..n_units {
                    col_delta_sq[u] += c_sq[u];
                }
            }
        });
    }

    let old_norm = old_norm_sq.sqrt();
    let new_norm = new_norm_sq.sqrt();
    let delta_norm = delta_norm_sq.sqrt();
    let denom = (old_norm * new_norm).max(1e-12);
    let overall_cos = dot_new_old / denom;

    // Identify changed columns (> 1e-6 L2 norm) and the top modified column
    let mut active_cols = Vec::new();
    let mut top_unit = 0usize;
    let mut max_col_sq = -1.0f64;
    for (u, &sq) in col_delta_sq.iter().enumerate() {
        if sq > 1e-12 {
            active_cols.push(u);
        }
        if sq > max_col_sq {
            max_col_sq = sq;
            top_unit = u;
        }
    }
    let changed_units_count = active_cols.len();

    // If surgical edit (0 < changed_units_count <= 64), extract old and new column for `top_unit`
    if changed_units_count > 0 && changed_units_count <= 64 && !out_old_col.is_null() && !out_new_col.is_null() {
        unsafe {
            let old_col_slice = slice::from_raw_parts_mut(out_old_col, hidden);
            let new_col_slice = slice::from_raw_parts_mut(out_new_col, hidden);
            for h in 0..hidden {
                let idx = if out_axis == 1 {
                    top_unit * hidden + h
                } else {
                    h * n_units + top_unit
                };
                old_col_slice[h] = read_elem(old_ptr, old_dtype, idx);
                new_col_slice[h] = read_elem(new_ptr, new_dtype, idx);
            }
        }
    }

    // Fast Principal Singular Vector `u0` and Singular Value `s0` of `dW_res` [hidden, n_units]
    let mut u0 = vec![0.0f64; hidden];
    let mut s0 = 0.0f64;

    if delta_norm > 1e-12 && !active_cols.is_empty() {
        if active_cols.len() <= 256 {
            // Materialize the compact [hidden, k] active delta submatrix in L1/L2 cache!
            let k = active_cols.len();
            let mut dw_sub = vec![0.0f64; hidden * k];
            for (ci, &u) in active_cols.iter().enumerate() {
                for h in 0..hidden {
                    let idx = if out_axis == 1 {
                        u * hidden + h
                    } else {
                        h * n_units + u
                    };
                    let vo = read_elem(old_ptr, old_dtype, idx) as f64;
                    let vn = read_elem(new_ptr, new_dtype, idx) as f64;
                    dw_sub[h * k + ci] = vn - vo;
                }
            }
            // Initialize v [k] proportional to column norms
            let mut v = vec![0.0f64; k];
            for (ci, &u) in active_cols.iter().enumerate() {
                v[ci] = col_delta_sq[u].sqrt();
            }
            for _ in 0..10 {
                // u0 = dw_sub * v
                let mut u_norm_sq = 0.0f64;
                for h in 0..hidden {
                    let row = &dw_sub[h * k..(h + 1) * k];
                    let mut sum = 0.0f64;
                    for ci in 0..k {
                        sum += row[ci] * v[ci];
                    }
                    u0[h] = sum;
                    u_norm_sq += sum * sum;
                }
                let u_norm = u_norm_sq.sqrt().max(1e-15);
                for h in 0..hidden {
                    u0[h] /= u_norm;
                }
                // v = dw_sub^T * u0
                let mut v_norm_sq = 0.0f64;
                for ci in 0..k {
                    v[ci] = 0.0;
                }
                for h in 0..hidden {
                    let uh = u0[h];
                    let row = &dw_sub[h * k..(h + 1) * k];
                    for ci in 0..k {
                        v[ci] += row[ci] * uh;
                    }
                }
                for ci in 0..k {
                    v_norm_sq += v[ci] * v[ci];
                }
                s0 = v_norm_sq.sqrt();
                let v_denom = s0.max(1e-15);
                for ci in 0..k {
                    v[ci] /= v_denom;
                }
            }
        } else {
            // Dense delta: initialize u0 from the top-norm delta column, then run 5 parallel power iterations
            for h in 0..hidden {
                let idx = if out_axis == 1 {
                    top_unit * hidden + h
                } else {
                    h * n_units + top_unit
                };
                u0[h] = (read_elem(new_ptr, new_dtype, idx) - read_elem(old_ptr, old_dtype, idx)) as f64;
            }
            let mut init_n = u0.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
            for x in u0.iter_mut() {
                *x /= init_n;
            }

            let mut v = vec![0.0f64; n_units];
            for _ in 0..5 {
                // Step A: v = dW^T * u0
                if out_axis == 1 {
                    let chunk_u = (n_units + n_threads - 1) / n_threads;
                    let u0_ref = &u0;
                    thread::scope(|s| {
                        let mut handles = Vec::new();
                        for (t, v_chunk) in v.chunks_mut(chunk_u).enumerate() {
                            let u_start = t * chunk_u;
                            handles.push(s.spawn(move || {
                                let op = old_addr as *const u8;
                                let np = new_addr as *const u8;
                                for (loc_i, slot) in v_chunk.iter_mut().enumerate() {
                                    let u = u_start + loc_i;
                                    let base = u * hidden;
                                    let mut acc = 0.0f64;
                                    for h in 0..hidden {
                                        let idx = base + h;
                                        let dv = (read_elem(np, new_dtype, idx) - read_elem(op, old_dtype, idx)) as f64;
                                        acc += dv * u0_ref[h];
                                    }
                                    *slot = acc;
                                }
                            }));
                        }
                        for h in handles {
                            h.join().unwrap();
                        }
                    });
                } else {
                    let chunk_h = (hidden + n_threads - 1) / n_threads;
                    let u0_ref = &u0;
                    v.fill(0.0);
                    thread::scope(|s| {
                        let mut handles = Vec::new();
                        for t in 0..n_threads {
                            let h_start = t * chunk_h;
                            let h_end = (h_start + chunk_h).min(hidden);
                            if h_start >= h_end {
                                continue;
                            }
                            handles.push(s.spawn(move || {
                                let op = old_addr as *const u8;
                                let np = new_addr as *const u8;
                                let mut loc_v = vec![0.0f64; n_units];
                                for h in h_start..h_end {
                                    let uh = u0_ref[h];
                                    let base = h * n_units;
                                    for u in 0..n_units {
                                        let idx = base + u;
                                        let dv = (read_elem(np, new_dtype, idx) - read_elem(op, old_dtype, idx)) as f64;
                                        loc_v[u] += dv * uh;
                                    }
                                }
                                loc_v
                            }));
                        }
                        for h in handles {
                            let lv = h.join().unwrap();
                            for u in 0..n_units {
                                v[u] += lv[u];
                            }
                        }
                    });
                }

                s0 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                let vd = s0.max(1e-15);
                for x in v.iter_mut() {
                    *x /= vd;
                }

                // Step B: u0 = dW * v
                if out_axis == 1 {
                    let chunk_u = (n_units + n_threads - 1) / n_threads;
                    let v_ref = &v;
                    u0.fill(0.0);
                    thread::scope(|s| {
                        let mut handles = Vec::new();
                        for t in 0..n_threads {
                            let u_start = t * chunk_u;
                            let u_end = (u_start + chunk_u).min(n_units);
                            if u_start >= u_end {
                                continue;
                            }
                            handles.push(s.spawn(move || {
                                let op = old_addr as *const u8;
                                let np = new_addr as *const u8;
                                let mut loc_u = vec![0.0f64; hidden];
                                for u in u_start..u_end {
                                    let vu = v_ref[u];
                                    let base = u * hidden;
                                    for h in 0..hidden {
                                        let idx = base + h;
                                        let dv = (read_elem(np, new_dtype, idx) - read_elem(op, old_dtype, idx)) as f64;
                                        loc_u[h] += dv * vu;
                                    }
                                }
                                loc_u
                            }));
                        }
                        for h in handles {
                            let lu = h.join().unwrap();
                            for i in 0..hidden {
                                u0[i] += lu[i];
                            }
                        }
                    });
                } else {
                    let chunk_h = (hidden + n_threads - 1) / n_threads;
                    let v_ref = &v;
                    thread::scope(|s| {
                        let mut handles = Vec::new();
                        for (t, u_chunk) in u0.chunks_mut(chunk_h).enumerate() {
                            let h_start = t * chunk_h;
                            handles.push(s.spawn(move || {
                                let op = old_addr as *const u8;
                                let np = new_addr as *const u8;
                                for (loc_i, slot) in u_chunk.iter_mut().enumerate() {
                                    let h = h_start + loc_i;
                                    let base = h * n_units;
                                    let mut acc = 0.0f64;
                                    for u in 0..n_units {
                                        let idx = base + u;
                                        let dv = (read_elem(np, new_dtype, idx) - read_elem(op, old_dtype, idx)) as f64;
                                        acc += dv * v_ref[u];
                                    }
                                    *slot = acc;
                                }
                            }));
                        }
                        for h in handles {
                            h.join().unwrap();
                        }
                    });
                }

                init_n = u0.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
                for x in u0.iter_mut() {
                    *x /= init_n;
                }
            }
        }
    }

    // Compute old_energy = ||u0^T W_old||_2 and new_energy = ||u0^T W_new||_2 in parallel
    let mut old_energy_sq = 0.0f64;
    let mut new_energy_sq = 0.0f64;
    let u0_ref = &u0;

    if out_axis == 1 {
        let chunk_u = (n_units + n_threads - 1) / n_threads;
        thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..n_threads {
                let u_start = t * chunk_u;
                let u_end = (u_start + chunk_u).min(n_units);
                if u_start >= u_end {
                    continue;
                }
                handles.push(s.spawn(move || {
                    let op = old_addr as *const u8;
                    let np = new_addr as *const u8;
                    let mut loc_oe = 0.0f64;
                    let mut loc_ne = 0.0f64;
                    for u in u_start..u_end {
                        let base = u * hidden;
                        let mut dot_o = 0.0f64;
                        let mut dot_n = 0.0f64;
                        for h in 0..hidden {
                            let idx = base + h;
                            let uh = u0_ref[h];
                            dot_o += (read_elem(op, old_dtype, idx) as f64) * uh;
                            dot_n += (read_elem(np, new_dtype, idx) as f64) * uh;
                        }
                        loc_oe += dot_o * dot_o;
                        loc_ne += dot_n * dot_n;
                    }
                    (loc_oe, loc_ne)
                }));
            }
            for h in handles {
                let (oe, ne) = h.join().unwrap();
                old_energy_sq += oe;
                new_energy_sq += ne;
            }
        });
    } else {
        let chunk_h = (hidden + n_threads - 1) / n_threads;
        let mut proj_o = vec![0.0f64; n_units];
        let mut proj_n = vec![0.0f64; n_units];
        thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..n_threads {
                let h_start = t * chunk_h;
                let h_end = (h_start + chunk_h).min(hidden);
                if h_start >= h_end {
                    continue;
                }
                handles.push(s.spawn(move || {
                    let op = old_addr as *const u8;
                    let np = new_addr as *const u8;
                    let mut loc_po = vec![0.0f64; n_units];
                    let mut loc_pn = vec![0.0f64; n_units];
                    for h in h_start..h_end {
                        let uh = u0_ref[h];
                        let base = h * n_units;
                        for u in 0..n_units {
                            let idx = base + u;
                            loc_po[u] += (read_elem(op, old_dtype, idx) as f64) * uh;
                            loc_pn[u] += (read_elem(np, new_dtype, idx) as f64) * uh;
                        }
                    }
                    (loc_po, loc_pn)
                }));
            }
            for h in handles {
                let (po, pn) = h.join().unwrap();
                for u in 0..n_units {
                    proj_o[u] += po[u];
                    proj_n[u] += pn[u];
                }
            }
        });
        old_energy_sq = proj_o.iter().map(|x| x * x).sum();
        new_energy_sq = proj_n.iter().map(|x| x * x).sum();
    }

    unsafe {
        let top_u_slice = slice::from_raw_parts_mut(out_top_u, hidden);
        for h in 0..hidden {
            top_u_slice[h] = (u0[h] * s0) as f32;
        }
        *out_stats = ResidualDeltaStats {
            old_norm,
            new_norm,
            delta_norm,
            overall_cosine_new_old: overall_cos,
            values_changed,
            changed_units_count: changed_units_count as u64,
            top_modified_unit: top_unit as u64,
            s0,
            old_energy: old_energy_sq.sqrt(),
            new_energy: new_energy_sq.sqrt(),
        };
    }

    0
}
