//! `wsi-bake`: zero-dependency standalone CLI for inspecting and baking `.moipatch` files.

mod json;
mod safetensors;
mod sha256;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use json::{parse_json, JsonValue};
use safetensors::{
    convert_bytes_for_sha256, read_elem_f64, write_elem_f64, DType, SafeTensorsFile, SafeTensorsHeader,
};
use sha256::sha256_hex;

struct PatchEdit {
    module: String,
    u: Vec<f64>,
    u_elem_bytes: usize,
    alpha: f64,
    idx: Option<Vec<usize>>,
    base_sha256: String,
    base_dtype: String,
    shape: Vec<usize>,
}

struct LoadedPatch {
    label: String,
    name: String,
    description: String,
    base_model: String,
    has_router: bool,
    router_layer: Option<u64>,
    router_thr: Option<f64>,
    has_veto: bool,
    veto_thr: Option<f64>,
    edits: Vec<PatchEdit>,
}

fn load_moipatch(path: &Path) -> Result<LoadedPatch, String> {
    let st = SafeTensorsFile::load(path)?;
    let raw_manifest = st
        .header
        .metadata
        .get("moipatch")
        .ok_or_else(|| format!("{}: missing 'moipatch' metadata", path.display()))?;
    let manifest = parse_json(raw_manifest)?;
    let fmt = manifest.get("format").and_then(|v| v.as_str()).unwrap_or("");
    if fmt != "moipatch/1" {
        return Err(format!("unsupported moipatch format '{fmt}' (expected 'moipatch/1')"));
    }

    let edits_arr = manifest
        .get("edits")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "missing 'edits' array in manifest".to_string())?;
    let mut edits = Vec::with_capacity(edits_arr.len());
    for (i, ev) in edits_arr.iter().enumerate() {
        let module = ev.get("module").and_then(|v| v.as_str()).ok_or_else(|| format!("edit {i}: missing module"))?.to_string();
        let alpha = ev.get("alpha").and_then(|v| v.as_f64()).ok_or_else(|| format!("edit {i}: missing alpha"))?;
        let has_idx = ev.get("has_idx").and_then(|v| v.as_bool()).unwrap_or(false);
        let base_sha256 = ev.get("base_sha256").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let base_dtype = ev.get("base_dtype").and_then(|v| v.as_str()).unwrap_or("bfloat16").to_string();
        let shape = ev
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_default();

        let (u_meta, _) = st.tensor_bytes(&format!("edit.{i}.u"))?;
        let (_, u) = st.tensor_to_f64(&format!("edit.{i}.u"))?;
        let idx = if has_idx {
            let raw_idx = st.tensor_to_i64(&format!("edit.{i}.idx"))?;
            Some(raw_idx.into_iter().map(|v| v as usize).collect())
        } else {
            None
        };
        edits.push(PatchEdit {
            module,
            u,
            u_elem_bytes: u_meta.dtype.elem_size(),
            alpha,
            idx,
            base_sha256,
            base_dtype,
            shape,
        });
    }

    let router_val = manifest.get("router").filter(|v| !v.is_null());
    let has_router = router_val.is_some();
    let router_layer = router_val.and_then(|r| r.get("layer")).and_then(|v| v.as_u64());
    let router_thr = router_val.and_then(|r| r.get("thr")).and_then(|v| v.as_f64());
    let has_veto = router_val.and_then(|r| r.get("veto_thr")).is_some()
        || st.header.tensors.contains_key("router.veto_dir");
    let veto_thr = router_val.and_then(|r| r.get("veto_thr")).and_then(|v| v.as_f64());

    let meta = manifest.get("meta");
    let label = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "patch".to_string());
    let name = meta.and_then(|m| m.get("name")).and_then(|v| v.as_str()).unwrap_or(&label).to_string();
    let description = meta.and_then(|m| m.get("description")).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let base_model = meta.and_then(|m| m.get("base_model")).and_then(|v| v.as_str()).unwrap_or("").to_string();

    Ok(LoadedPatch {
        label,
        name,
        description,
        base_model,
        has_router,
        router_layer,
        router_thr,
        has_veto,
        veto_thr,
        edits,
    })
}

/// Match Python's `re.search(r"(layers\.\d+\..+)$", name)` on keys ending with `.weight`
/// while ignoring `mtp` and `visual` tensors.
fn module_suffix_from_weight_key(tensor_key: &str) -> Option<String> {
    let base = tensor_key.strip_suffix(".weight")?;
    if base.contains("mtp") || base.contains("visual") {
        return None;
    }
    let bytes = base.as_bytes();
    let mut search_from = 0usize;
    while let Some(rel) = base[search_from..].find("layers.") {
        let abs = search_from + rel;
        let boundary_ok = abs == 0 || bytes[abs - 1] == b'.';
        let after = &base[abs + 7..];
        let digits = after.bytes().take_while(|b| b.is_ascii_digit()).count();
        if boundary_ok && digits > 0 && after.as_bytes().get(digits) == Some(&b'.') {
            return Some(base[abs..].to_string());
        }
        search_from = abs + 7;
    }
    None
}

fn index_checkpoint_shards(model_dir: &Path) -> Result<(Vec<String>, BTreeMap<String, Vec<(String, String)>>), String> {
    let idx_path = model_dir.join("model.safetensors.index.json");
    let mut suffix_map: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    let mut shard_names = BTreeSet::new();

    if idx_path.exists() {
        let text = fs::read_to_string(&idx_path).map_err(|e| format!("failed reading {}: {e}", idx_path.display()))?;
        let val = parse_json(&text)?;
        if let Some(wmap) = val.get("weight_map").and_then(|w| w.as_object()) {
            for (k, fname_v) in wmap {
                let Some(fname) = fname_v.as_str() else { continue };
                shard_names.insert(fname.to_string());
                if let Some(suf) = module_suffix_from_weight_key(k) {
                    suffix_map.entry(suf).or_default().push((fname.to_string(), k.clone()));
                }
            }
        }
    } else {
        let entries = fs::read_dir(model_dir).map_err(|e| format!("failed listing {}: {e}", model_dir.display()))?;
        let mut candidates = Vec::new();
        for ent in entries.flatten() {
            let name = ent.file_name().to_string_lossy().to_string();
            if name.ends_with(".safetensors") && !name.starts_with("edit_") && name != "delta_lowrank.safetensors" {
                candidates.push(name);
            }
        }
        candidates.sort();
        for fname in candidates {
            let hdr: SafeTensorsHeader = SafeTensorsFile::read_header(&model_dir.join(&fname))?;
            shard_names.insert(fname.clone());
            for k in hdr.tensors.keys() {
                if let Some(suf) = module_suffix_from_weight_key(k) {
                    suffix_map.entry(suf).or_default().push((fname.clone(), k.clone()));
                }
            }
        }
    }

    if shard_names.is_empty() {
        return Err(format!("no safetensors checkpoint files found in {}", model_dir.display()));
    }
    Ok((shard_names.into_iter().collect(), suffix_map))
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    fs::create_dir_all(dst).map_err(|e| format!("failed mkdir {}: {e}", dst.display()))?;
    for ent in fs::read_dir(src).map_err(|e| format!("failed reading dir {}: {e}", src.display()))? {
        let ent = ent.map_err(|e| format!("dir entry error: {e}"))?;
        let p = ent.path();
        let out_p = dst.join(ent.file_name());
        if p.is_dir() {
            copy_dir_recursive(&p, &out_p)?;
        } else {
            fs::copy(&p, &out_p).map_err(|e| format!("failed copying {}: {e}", p.display()))?;
        }
    }
    Ok(())
}

fn link_or_copy(src: &Path, dst: &Path, copy_only: bool) -> Result<(), String> {
    if dst.exists() {
        let _ = fs::remove_file(dst);
    }
    if !copy_only && fs::hard_link(src, dst).is_ok() {
        return Ok(());
    }
    fs::copy(src, dst).map_err(|e| format!("failed copying {}: {e}", src.display()))?;
    Ok(())
}

fn apply_edit_in_place(bytes: &mut [u8], dtype: DType, rows: usize, cols: usize, edit: &PatchEdit) -> Result<(), String> {
    if edit.u.len() != rows {
        return Err(format!(
            "{}: expected u length {rows}, got {}",
            edit.module,
            edit.u.len()
        ));
    }
    let all_cols: Vec<usize>;
    let col_indices: &[usize] = match &edit.idx {
        Some(v) => v.as_slice(),
        None => {
            all_cols = (0..cols).collect();
            all_cols.as_slice()
        }
    };

    for &j in col_indices {
        if j >= cols {
            return Err(format!("{}: column index {j} out of bounds ({cols})", edit.module));
        }
        let mut dot = 0.0f64;
        for (i, &u_i) in edit.u.iter().enumerate() {
            dot += u_i * read_elem_f64(bytes, dtype, i * cols + j);
        }
        if dot == 0.0 || edit.alpha == 0.0 {
            continue;
        }
        for (i, &u_i) in edit.u.iter().enumerate() {
            let off = i * cols + j;
            let w_ij = read_elem_f64(bytes, dtype, off);
            write_elem_f64(bytes, dtype, off, w_ij - edit.alpha * (u_i * dot));
        }
    }
    Ok(())
}

fn cmd_info(patch_path: &Path) -> Result<(), String> {
    let p = load_moipatch(patch_path)?;
    println!("{} [moipatch/1]", p.name);
    if !p.description.is_empty() {
        println!("{}", p.description);
    }
    if !p.base_model.is_empty() {
        println!("base model: {}", p.base_model);
    }
    let mut payload_bytes = 0usize;
    for e in &p.edits {
        let cols_desc = match &e.idx {
            Some(v) => format!("{} columns", v.len()),
            None => "all columns".to_string(),
        };
        let sha_pref = &e.base_sha256[..e.base_sha256.len().min(12)];
        let shp_str = if e.shape.len() == 2 {
            format!("({}, {})", e.shape[0], e.shape[1])
        } else {
            format!("{:?}", e.shape)
        };
        println!(
            "  edit {} {} alpha {}, {}, base {} sha256 {}...",
            e.module, shp_str, e.alpha, cols_desc, e.base_dtype, sha_pref
        );
        payload_bytes += e.u.len() * e.u_elem_bytes + e.idx.as_ref().map(|v| v.len() * 8).unwrap_or(0);
    }
    if p.has_router {
        println!(
            "  router: hidden_states[{}] last token . dir > {:.4}",
            p.router_layer.unwrap_or(0),
            p.router_thr.unwrap_or(0.0)
        );
        if p.has_veto {
            println!(
                "  router veto: hidden_states[{}] last token . veto_dir <= {:.4}",
                p.router_layer.unwrap_or(0),
                p.veto_thr.unwrap_or(0.0)
            );
        }
    }
    println!("  payload: {:.1} KB of recipe", (payload_bytes as f64) / 1024.0);
    Ok(())
}

fn cmd_export(args: &[String]) -> Result<(), String> {
    let mut patch_paths: Vec<PathBuf> = Vec::new();
    let mut model_dir: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut allow_routed = false;
    let mut copy_only = false;
    let mut verify_fp = true;
    let mut as_json = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_dir = Some(PathBuf::from(args.get(i).ok_or("--model requires a path")?));
            }
            "--out" => {
                i += 1;
                out_dir = Some(PathBuf::from(args.get(i).ok_or("--out requires a path")?));
            }
            "--force" => allow_routed = true,
            "--copy" => copy_only = true,
            "--no-verify" => verify_fp = false,
            "--json" => as_json = true,
            flag if flag.starts_with('-') => return Err(format!("unknown flag '{flag}'")),
            pos => patch_paths.push(PathBuf::from(pos)),
        }
        i += 1;
    }

    let mdir = model_dir.ok_or("missing required --model <dir>")?;
    let odir = out_dir.ok_or("missing required --out <dir>")?;
    if patch_paths.is_empty() {
        return Err("provide at least one .moipatch file".to_string());
    }
    if !mdir.exists() {
        return Err(format!("base model directory not found: {}", mdir.display()));
    }
    if fs::canonicalize(&mdir).ok() == fs::canonicalize(&odir).ok() && odir.exists() {
        return Err("out_dir must differ from model_dir".to_string());
    }

    let mut patches = Vec::with_capacity(patch_paths.len());
    for p in &patch_paths {
        patches.push(load_moipatch(p)?);
    }

    let routed_labels: Vec<String> = patches.iter().filter(|p| p.has_router).map(|p| p.label.clone()).collect();
    if !routed_labels.is_empty() && !allow_routed {
        return Err(format!(
            "Patch(es) {:?} contain a runtime router/safety veto (`router`). Pass --force (allow_routed=True) to bake statically as always_on.",
            routed_labels
        ));
    }

    let (shards, suffix_map) = index_checkpoint_shards(&mdir)?;
    let shard_set: BTreeSet<String> = shards.iter().cloned().collect();
    let mut shard_ops: BTreeMap<String, Vec<(String, String, &PatchEdit)>> = BTreeMap::new();
    for p in &patches {
        for e in &p.edits {
            let hits = suffix_map.get(&e.module).map(|v| v.as_slice()).unwrap_or(&[]);
            if hits.len() != 1 {
                return Err(format!("{}: expected 1 weight tensor in {}, found {}", e.module, mdir.display(), hits.len()));
            }
            let (shard_rel, tensor_key) = &hits[0];
            shard_ops
                .entry(shard_rel.clone())
                .or_default()
                .push((p.label.clone(), tensor_key.clone(), e));
        }
    }

    fs::create_dir_all(&odir).map_err(|e| format!("failed creating {}: {e}", odir.display()))?;
    let touched_names: BTreeSet<String> = shard_ops.keys().cloned().collect();

    let mut dir_entries: Vec<_> = fs::read_dir(&mdir)
        .map_err(|e| format!("failed reading {}: {e}", mdir.display()))?
        .filter_map(|e| e.ok())
        .collect();
    dir_entries.sort_by_key(|e| e.file_name());
    for ent in dir_entries {
        let name = ent.file_name().to_string_lossy().to_string();
        if name == ".git" || name == "__pycache__" {
            continue;
        }
        let src = ent.path();
        let dst = odir.join(&name);
        if src.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else if touched_names.contains(&name) {
            continue;
        } else if shard_set.contains(&name) {
            link_or_copy(&src, &dst, copy_only)?;
        } else {
            fs::copy(&src, &dst).map_err(|e| format!("failed copying {}: {e}", src.display()))?;
        }
    }

    struct BakedModInfo {
        shard: String,
        tensor_key: String,
        base_sha256: String,
        patches: Vec<(String, f64)>,
        baked_sha256: String,
    }
    let mut baked_modules: BTreeMap<String, BakedModInfo> = BTreeMap::new();
    let mut verified_keys: BTreeSet<String> = BTreeSet::new();

    for (shard_rel, ops) in &shard_ops {
        let mut sf = SafeTensorsFile::load(&mdir.join(shard_rel))?;
        for (lbl, tensor_key, edit) in ops {
            let (meta, bytes) = sf.tensor_bytes_mut(tensor_key)?;
            if meta.shape.len() != 2 {
                return Err(format!("{tensor_key}: expected 2D matrix, got {:?}", meta.shape));
            }
            if !verified_keys.contains(tensor_key) {
                let fp_dtype = DType::from_name(&edit.base_dtype).unwrap_or(meta.dtype);
                let orig_sha = if fp_dtype == meta.dtype {
                    sha256_hex(bytes)
                } else {
                    sha256_hex(&convert_bytes_for_sha256(bytes, meta.dtype, fp_dtype)?)
                };
                if verify_fp && !edit.base_sha256.is_empty() && orig_sha != edit.base_sha256 {
                    return Err(format!(
                        "{} in {}: base SHA-256 mismatch ({} != {})",
                        edit.module,
                        shard_rel,
                        &orig_sha[..orig_sha.len().min(12)],
                        &edit.base_sha256[..edit.base_sha256.len().min(12)],
                    ));
                }
                verified_keys.insert(tensor_key.clone());
                baked_modules.insert(
                    edit.module.clone(),
                    BakedModInfo {
                        shard: shard_rel.clone(),
                        tensor_key: tensor_key.clone(),
                        base_sha256: orig_sha,
                        patches: Vec::new(),
                        baked_sha256: String::new(),
                    },
                );
            }

            apply_edit_in_place(bytes, meta.dtype, meta.shape[0], meta.shape[1], edit)?;
            let new_sha = sha256_hex(bytes);
            if let Some(info) = baked_modules.get_mut(&edit.module) {
                info.patches.push((lbl.clone(), edit.alpha));
                info.baked_sha256 = new_sha;
            }
        }
        sf.save_atomic(&odir.join(shard_rel))?;
    }

    let patches_json: Vec<JsonValue> = patches
        .iter()
        .map(|p| {
            JsonValue::obj(vec![
                ("label", JsonValue::String(p.label.clone())),
                ("name", JsonValue::String(p.name.clone())),
                ("has_router", JsonValue::Bool(p.has_router)),
            ])
        })
        .collect();
    let mut mods_map = BTreeMap::new();
    for (mod_name, info) in &baked_modules {
        let plist = info
            .patches
            .iter()
            .map(|(plbl, a)| {
                JsonValue::obj(vec![
                    ("patch", JsonValue::String(plbl.clone())),
                    ("alpha", JsonValue::num_f64(*a)),
                ])
            })
            .collect();
        mods_map.insert(
            mod_name.clone(),
            JsonValue::obj(vec![
                ("shard", JsonValue::String(info.shard.clone())),
                ("tensor_key", JsonValue::String(info.tensor_key.clone())),
                ("base_sha256", JsonValue::String(info.base_sha256.clone())),
                ("patches", JsonValue::Array(plist)),
                ("baked_sha256", JsonValue::String(info.baked_sha256.clone())),
            ]),
        );
    }
    let provenance = JsonValue::obj(vec![
        ("base_model_dir", JsonValue::String(mdir.to_string_lossy().to_string())),
        ("out_dir", JsonValue::String(odir.to_string_lossy().to_string())),
        ("patches", JsonValue::Array(patches_json)),
        ("had_router_warning", JsonValue::Bool(!routed_labels.is_empty())),
        (
            "routed_patches",
            JsonValue::Array(routed_labels.iter().map(|s| JsonValue::String(s.clone())).collect()),
        ),
        ("n_total_shards", JsonValue::num_u64(shards.len() as u64)),
        ("n_touched_shards", JsonValue::num_u64(touched_names.len() as u64)),
        ("modules", JsonValue::Object(mods_map)),
    ]);

    let prov_bytes = provenance.to_json_string(true);
    let prov_tmp = odir.join("wsi_baked_patches.json.tmp");
    let prov_dst = odir.join("wsi_baked_patches.json");
    if prov_dst.exists() {
        let _ = fs::remove_file(&prov_dst);
    }
    fs::write(&prov_tmp, prov_bytes.as_bytes()).map_err(|e| format!("failed writing provenance: {e}"))?;
    fs::rename(&prov_tmp, &prov_dst).map_err(|e| format!("failed renaming provenance: {e}"))?;

    if as_json {
        println!("{prov_bytes}");
    } else {
        println!(
            "Baked {} patch(es) into {} ({}/{} shard(s) modified, {} module(s))",
            patches.len(),
            odir.display(),
            touched_names.len(),
            shards.len(),
            baked_modules.len()
        );
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help") {
        println!("Usage:\n  wsi-bake info <patch.moipatch>\n  wsi-bake export <patch...> --model <dir> --out <dir> [--force] [--copy] [--no-verify] [--json]");
        return;
    }

    let res = match args[0].as_str() {
        "info" => {
            if args.len() < 2 {
                Err("usage: wsi-bake info <patch.moipatch>".to_string())
            } else {
                cmd_info(Path::new(&args[1]))
            }
        }
        "export" => cmd_export(&args[1..]),
        other => Err(format!("unknown subcommand '{other}' (expected 'info' or 'export')")),
    };

    if let Err(err) = res {
        eprintln!("wsi-bake error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::module_suffix_from_weight_key;

    #[test]
    fn suffix_matches_python() {
        assert_eq!(
            module_suffix_from_weight_key("model.layers.15.feed_forward.w2.weight").as_deref(),
            Some("layers.15.feed_forward.w2")
        );
        assert_eq!(module_suffix_from_weight_key("model.mtp.layers.0.mlp.weight"), None);
        assert_eq!(module_suffix_from_weight_key("visual.blocks.0.layers.1.mlp.weight"), None);
        assert_eq!(module_suffix_from_weight_key("lm_head.weight"), None);
    }
}
