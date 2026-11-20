"""restart_control.report: resumed answers are graded from prefix + continuation, unrouted ones fall back to base."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import restart_control as rc  # noqa: E402


class CharTok:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i)


def ids(s):
    return [ord(c) for c in s]


def _write(work, name, rows):
    (work / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("dataset", ["math", "gsm8k"])
def test_report_pairs_restart_against_patched_moi(tmp_path, dataset):
    ans = (lambda v: f"so \\boxed{{{v}}}") if dataset == "math" else (lambda v: f"so #### {v}")
    (tmp_path / "config.json").write_text(json.dumps({"base_dir": "x", "eos": [0], "model_repo": "toy"}))
    prompt = ids("Q:")
    n = 6
    _write(tmp_path, "requests_test.jsonl",
           [{"id": i, "prompt_ids": prompt, "max_tokens": 200, "gold": "4", "seed": i} for i in range(n)])
    # base: all wrong; the router fires on 0..3 (cut after "work ")
    _write(tmp_path, "outputs_test_base.jsonl",
           [{"id": i, "token_ids": ids("work " + ans(9)) + [0], "finish_reason": "stop"} for i in range(n)])
    pre = ids("work ")
    _write(tmp_path, rc.PLAN, [{"id": i, "prompt_ids": prompt + pre, "max_tokens": 195, "seed": i + 7}
                               for i in range(4)])

    def cont(right):
        return [{"id": i, "token_ids": ids(ans(4 if i in right else 7)) + [0], "finish_reason": "stop"}
                for i in range(4)]

    moi = rc.MOI_ARMS[dataset]
    _write(tmp_path, moi["wsi_moi"], cont({0, 1, 2}))
    _write(tmp_path, moi["random_moi"], cont({0}))
    _write(tmp_path, rc.RESTART_OUT, cont({0, 1, 2}))

    res = rc.report(tmp_path, dataset, tok=CharTok())
    assert res["router_fired"] == pytest.approx(4 / 6)
    assert res["arms"]["base"]["correct"] == 0
    assert res["arms"]["restart_moi"]["correct"] == pytest.approx(3 / 6)
    assert res["arms"]["random_moi"]["correct"] == pytest.approx(1 / 6, abs=1e-3)
    p = res["paired"]["wsi_moi_vs_restart_moi"]
    assert p["diff"] == 0 and p["a_only"] == p["b_only"] == 0
    assert res["paired"]["restart_moi_vs_base"]["a_only"] == 3
    assert res["verdict"].startswith("edit ties restart")
