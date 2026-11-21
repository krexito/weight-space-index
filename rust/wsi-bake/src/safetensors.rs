//! Minimal zero-dependency safetensors reader/writer and IEEE-754 dtype conversions.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::json::parse_json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    BF16,
    F16,
    F32,
    F64,
    I64,
    I32,
    U8,
    I8,
    U16,
    I16,
    BOOL,
}

impl DType {
    pub fn from_name(s: &str) -> Result<Self, String> {
        match s {
            "BF16" | "bfloat16" => Ok(DType::BF16),
            "F16" | "float16" => Ok(DType::F16),
            "F32" | "float32" => Ok(DType::F32),
            "F64" | "float64" => Ok(DType::F64),
            "I64" | "int64" => Ok(DType::I64),
            "I32" | "int32" => Ok(DType::I32),
            "U8" | "uint8" => Ok(DType::U8),
            "I8" | "int8" => Ok(DType::I8),
            "U16" | "uint16" => Ok(DType::U16),
            "I16" | "int16" => Ok(DType::I16),
            "BOOL" | "bool" => Ok(DType::BOOL),
            other => Err(format!("unsupported safetensors dtype '{other}'")),
        }
    }

    pub fn elem_size(self) -> usize {
        match self {
            DType::BF16 | DType::F16 | DType::U16 | DType::I16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::U8 | DType::I8 | DType::BOOL => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct SafeTensorsHeader {
    pub metadata: BTreeMap<String, String>,
    pub tensors: BTreeMap<String, TensorMeta>,
    pub data_offset: usize,
}

pub struct SafeTensorsFile {
    pub header: SafeTensorsHeader,
    pub raw: Vec<u8>,
}

impl SafeTensorsFile {
    pub fn read_header(path: &Path) -> Result<SafeTensorsHeader, String> {
        use std::io::Read as _;
        let mut f = fs::File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf).map_err(|e| format!("failed reading header length in {}: {e}", path.display()))?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        let mut hdr_bytes = vec![0u8; header_len];
        f.read_exact(&mut hdr_bytes).map_err(|e| format!("failed reading header JSON in {}: {e}", path.display()))?;
        parse_header_json(&hdr_bytes, 8 + header_len)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = fs::read(path).map_err(|e| format!("failed reading {}: {e}", path.display()))?;
        if raw.len() < 8 {
            return Err(format!("safetensors file too short: {}", path.display()));
        }
        let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let data_offset = 8 + header_len;
        if data_offset > raw.len() {
            return Err(format!("invalid header length in {}", path.display()));
        }
        let header = parse_header_json(&raw[8..data_offset], data_offset)?;
        Ok(Self { header, raw })
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<(&TensorMeta, &[u8]), String> {
        let meta = self.header.tensors.get(name).ok_or_else(|| format!("tensor '{name}' not found"))?;
        let start = self.header.data_offset + meta.start;
        let end = self.header.data_offset + meta.end;
        if end > self.raw.len() {
            return Err(format!("tensor '{name}' offsets out of bounds"));
        }
        Ok((meta, &self.raw[start..end]))
    }

    pub fn tensor_bytes_mut(&mut self, name: &str) -> Result<(TensorMeta, &mut [u8]), String> {
        let meta = self.header.tensors.get(name).ok_or_else(|| format!("tensor '{name}' not found"))?.clone();
        let start = self.header.data_offset + meta.start;
        let end = self.header.data_offset + meta.end;
        if end > self.raw.len() {
            return Err(format!("tensor '{name}' offsets out of bounds"));
        }
        Ok((meta, &mut self.raw[start..end]))
    }

    pub fn tensor_to_f64(&self, name: &str) -> Result<(Vec<usize>, Vec<f64>), String> {
        let (meta, bytes) = self.tensor_bytes(name)?;
        let n = bytes.len() / meta.dtype.elem_size();
        let mut out = Vec::with_capacity(n);
        match meta.dtype {
            DType::F64 => {
                for ch in bytes.chunks_exact(8) {
                    out.push(f64::from_le_bytes(ch.try_into().unwrap()));
                }
            }
            DType::F32 => {
                for ch in bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes(ch.try_into().unwrap()) as f64);
                }
            }
            DType::BF16 => {
                for ch in bytes.chunks_exact(2) {
                    out.push(bf16_to_f64(u16::from_le_bytes([ch[0], ch[1]])));
                }
            }
            DType::F16 => {
                for ch in bytes.chunks_exact(2) {
                    out.push(f16_to_f64(u16::from_le_bytes([ch[0], ch[1]])));
                }
            }
            other => return Err(format!("cannot convert '{name}' ({other:?}) to f64")),
        }
        Ok((meta.shape.clone(), out))
    }

    pub fn tensor_to_i64(&self, name: &str) -> Result<Vec<i64>, String> {
        let (meta, bytes) = self.tensor_bytes(name)?;
        match meta.dtype {
            DType::I64 => Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()),
            DType::I32 => Ok(bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect()),
            other => Err(format!("expected integer tensor for '{name}', got {other:?}")),
        }
    }

    pub fn save_atomic(&self, dst: &Path) -> Result<(), String> {
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("failed to create parent {}: {e}", parent.display()))?;
        }
        if dst.exists() {
            let _ = fs::remove_file(dst);
        }
        let tmp = dst.with_file_name(format!("{}.tmp", dst.file_name().unwrap_or_default().to_string_lossy()));
        fs::write(&tmp, &self.raw).map_err(|e| format!("failed writing {}: {e}", tmp.display()))?;
        fs::rename(&tmp, dst).map_err(|e| format!("failed renaming to {}: {e}", dst.display()))?;
        Ok(())
    }
}

fn parse_header_json(bytes: &[u8], data_offset: usize) -> Result<SafeTensorsHeader, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "safetensors header is not UTF-8".to_string())?;
    let val = parse_json(text)?;
    let obj = val.as_object().ok_or_else(|| "safetensors header is not a JSON object".to_string())?;

    let mut metadata = BTreeMap::new();
    let mut tensors = BTreeMap::new();
    for (k, v) in obj {
        if k == "__metadata__" {
            if let Some(m) = v.as_object() {
                for (mk, mv) in m {
                    if let Some(ms) = mv.as_str() {
                        metadata.insert(mk.clone(), ms.to_string());
                    }
                }
            }
            continue;
        }
        let raw_dtype = v.get("dtype").and_then(|x| x.as_str()).ok_or_else(|| format!("missing dtype for {k}"))?.to_string();
        let dtype = DType::from_name(&raw_dtype)?;
        let shape_arr = v.get("shape").and_then(|x| x.as_array()).ok_or_else(|| format!("missing shape for {k}"))?;
        let mut shape = Vec::with_capacity(shape_arr.len());
        for s in shape_arr {
            shape.push(s.as_u64().ok_or_else(|| format!("invalid shape dim in {k}"))? as usize);
        }
        let offs = v.get("data_offsets").and_then(|x| x.as_array()).ok_or_else(|| format!("missing offsets for {k}"))?;
        if offs.len() != 2 {
            return Err(format!("invalid data_offsets in {k}"));
        }
        let start = offs[0].as_u64().ok_or_else(|| format!("invalid start offset in {k}"))? as usize;
        let end = offs[1].as_u64().ok_or_else(|| format!("invalid end offset in {k}"))? as usize;
        tensors.insert(k.clone(), TensorMeta { dtype, shape, start, end });
    }
    Ok(SafeTensorsHeader { metadata, tensors, data_offset })
}

#[inline]
pub fn bf16_to_f64(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

#[inline]
pub fn f64_to_bf16(val: f64) -> u16 {
    // Match PyTorch CPU: cast f64 -> f32 round-to-nearest-even, then f32 -> bf16 round-to-nearest-even.
    let f = val as f32;
    let bits = f.to_bits();
    let exp = bits & 0x7f80_0000;
    let mant = bits & 0x007f_ffff;
    if exp == 0x7f80_0000 {
        if mant != 0 {
            return ((bits >> 16) as u16) | 0x0040; // quiet NaN
        }
        return (bits >> 16) as u16; // Inf
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

#[inline]
pub fn f16_to_f64(bits: u16) -> f64 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let mant = (bits & 0x03ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e: i32 = 127 - 15 + 1;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            sign | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        let e = (exp as u32) + (127 - 15);
        sign | (e << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits) as f64
}

#[inline]
pub fn f64_to_f16(val: f64) -> u16 {
    // Match PyTorch CPU: cast f64 -> f32 round-to-nearest-even, then f32 -> IEEE f16 round-to-nearest-even.
    let f = val as f32;
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 255 {
        if mant != 0 {
            let nan_m = ((mant >> 13) as u16) | 0x0200;
            return sign | 0x7c00 | nan_m;
        }
        return sign | 0x7c00;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00; // overflow -> Inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign; // underflow -> signed zero
        }
        // Subnormal f16
        let full_mant = mant | 0x0080_0000;
        let shift = (14 - new_exp) as u32; // >= 14 and <= 24
        let half_mant = full_mant >> shift;
        let rem = full_mant & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let round_up = rem > halfway || (rem == halfway && (half_mant & 1) == 1);
        let rounded = half_mant + if round_up { 1 } else { 0 };
        return sign | (rounded as u16);
    }

    let half_mant = mant >> 13;
    let rem = mant & 0x1fff;
    let round_up = rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) == 1);
    let rounded_mant = half_mant + if round_up { 1 } else { 0 };
    if rounded_mant == 0x0400 {
        let next_exp = new_exp + 1;
        if next_exp >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((next_exp as u16) << 10);
    }
    sign | ((new_exp as u16) << 10) | (rounded_mant as u16)
}

#[inline]
pub fn read_elem_f64(bytes: &[u8], dtype: DType, elem_idx: usize) -> f64 {
    match dtype {
        DType::BF16 => {
            let i = elem_idx * 2;
            bf16_to_f64(u16::from_le_bytes([bytes[i], bytes[i + 1]]))
        }
        DType::F16 => {
            let i = elem_idx * 2;
            f16_to_f64(u16::from_le_bytes([bytes[i], bytes[i + 1]]))
        }
        DType::F32 => {
            let i = elem_idx * 4;
            f32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as f64
        }
        DType::F64 => {
            let i = elem_idx * 8;
            f64::from_le_bytes(bytes[i..i + 8].try_into().unwrap())
        }
        _ => unreachable!("unsupported weight dtype in read_elem_f64"),
    }
}

#[inline]
pub fn write_elem_f64(bytes: &mut [u8], dtype: DType, elem_idx: usize, val: f64) {
    match dtype {
        DType::BF16 => {
            let i = elem_idx * 2;
            let b = f64_to_bf16(val).to_le_bytes();
            bytes[i] = b[0];
            bytes[i + 1] = b[1];
        }
        DType::F16 => {
            let i = elem_idx * 2;
            let b = f64_to_f16(val).to_le_bytes();
            bytes[i] = b[0];
            bytes[i + 1] = b[1];
        }
        DType::F32 => {
            let i = elem_idx * 4;
            let b = (val as f32).to_le_bytes();
            bytes[i..i + 4].copy_from_slice(&b);
        }
        DType::F64 => {
            let i = elem_idx * 8;
            let b = val.to_le_bytes();
            bytes[i..i + 8].copy_from_slice(&b);
        }
        _ => unreachable!("unsupported weight dtype in write_elem_f64"),
    }
}

pub fn convert_bytes_for_sha256(bytes: &[u8], src_dtype: DType, target_dtype: DType) -> Result<Vec<u8>, String> {
    if src_dtype == target_dtype {
        return Ok(bytes.to_vec());
    }
    let n = bytes.len() / src_dtype.elem_size();
    let mut out = vec![0u8; n * target_dtype.elem_size()];
    for i in 0..n {
        let v = read_elem_f64(bytes, src_dtype, i);
        write_elem_f64(&mut out, target_dtype, i, v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{bf16_to_f64, f16_to_f64, f64_to_bf16, f64_to_f16};

    #[test]
    fn bf16_and_f16_round_to_nearest_even() {
        assert_eq!(f64_to_bf16(1.0 + 2.0f64.powi(-8)), 0x3f80);
        assert_eq!(f64_to_bf16(1.0 + 3.0 * 2.0f64.powi(-8)), 0x3f82);
        assert_eq!(f64_to_bf16(1.0 + 2.0f64.powi(-8) + 2.0f64.powi(-52)), 0x3f80);
        assert!((bf16_to_f64(0x3f80) - 1.0).abs() == 0.0);

        assert_eq!(f64_to_f16(1.0 + 2.0f64.powi(-11)), 0x3c00);
        assert_eq!(f64_to_f16(1.0 + 3.0 * 2.0f64.powi(-11)), 0x3c02);
        assert_eq!(f64_to_f16(1.0 + 2.0f64.powi(-11) + 2.0f64.powi(-52)), 0x3c00);
        assert!((f16_to_f64(0x3c00) - 1.0).abs() == 0.0);
        assert_eq!(f64_to_f16(-0.5), 0xb800);
    }
}
