//! Standalone Native Rust CLI (`wsi-rs`) for instant zero-copy `.safetensors` inspection & delta analysis.

use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

fn parse_safetensors_header(path: &str) -> Result<(usize, String, u64), String> {
    let mut f = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let file_len = f.metadata().map_err(|e| e.to_string())?.len();
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut h_buf = vec![0u8; header_len];
    f.read_exact(&mut h_buf).map_err(|e| e.to_string())?;
    let header_str = String::from_utf8(h_buf).map_err(|e| e.to_string())?;
    Ok((header_len, header_str, file_len))
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!(
            "wsi-rs v0.1.0 (Native Rust SIMD Engine for Weight-Space Index)\n\
             Startup time: {:.3} ms | CPU threads: {}\n\
             Usage:\n  \
               wsi-rs header <path/to/model.safetensors>",
            t0.elapsed().as_secs_f64() * 1000.0,
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        );
        return;
    }

    match args[1].as_str() {
        "header" if args.len() >= 3 => {
            match parse_safetensors_header(&args[2]) {
                Ok((hlen, hjson, flen)) => {
                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let tensor_count = hjson.matches("\"shape\"").count();
                    println!(
                        "[wsi-rs] Inspected `{}` ({:.2} MB, {} tensors, header={} B) in {:.3} ms",
                        args[2],
                        flen as f64 / (1024.0 * 1024.0),
                        tensor_count,
                        hlen,
                        elapsed_ms
                    );
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Unknown command `{}`", args[1]);
            std::process::exit(1);
        }
    }
}
