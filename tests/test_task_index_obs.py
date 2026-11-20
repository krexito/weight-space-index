"""CPU smoke test: experiment_moi_task_index_lfm3.py with the pairwise index (math-obs+recon) on a tiny random model."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TOK = ROOT / "models" / "qwen35-0.8b-full"


@pytest.mark.skipif(not (TOK / "tokenizer_config.json").exists(), reason="needs a local Qwen tokenizer")
def test_obs_recon_index_runs_end_to_end(tmp_path):
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(TOK)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=40, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                      pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    mdir = tmp_path / "model"
    LlamaForCausalLM(cfg).save_pretrained(mdir)
    tok.save_pretrained(mdir)
    data = tmp_path / "data"
    (data / "gsm8k").mkdir(parents=True)
    (data / "wikitext").mkdir(parents=True)
    qa = pd.DataFrame({"question": [f"What is {i} plus {i}?" for i in range(12)],
                       "answer": [f"{i} + {i} = <<{i}+{i}={2 * i}>>{2 * i}\n#### {2 * i}" for i in range(12)]})
    qa.to_parquet(data / "gsm8k" / "train.parquet")
    qa.iloc[:4].to_parquet(data / "gsm8k" / "test.parquet")
    wt = pd.DataFrame({"text": ["The quick brown fox jumps over the lazy dog. " * 20] * 6})
    wt.to_parquet(data / "wikitext" / "validation.parquet")
    wt.to_parquet(data / "wikitext" / "test.parquet")
    out = tmp_path / "r.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "experiment_moi_task_index_lfm3.py"),
                        "--model-dir", str(mdir), "--data-dir", str(data), "--device", "cpu",
                        "--n-problems", "4", "--batch", "2", "--max-new-tokens", "4", "--budgets", "0.5,0.25",
                        "--indexes", "math-energy+recon,math-obs,math-obs+recon", "--calib-tokens", "600",
                        "--gram-tokens", "400", "--ppl-tokens", "300", "--out", str(out)],
                       capture_output=True, text=True, env=env, timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    r = json.loads(out.read_text())["results"]
    for b in ("0.5", "0.25"):
        a, o = r[f"math-energy+recon@{b}"], r[f"math-obs+recon@{b}"]
        assert o["neurons_per_layer"] == a["neurons_per_layer"] == [round(float(b) * 40)] * 2
        assert len(o["gsm8k"]["per_problem"]) == 4
    assert "obs selection for budgets [10, 20]" in p.stdout  # one greedy pass for both budgets
