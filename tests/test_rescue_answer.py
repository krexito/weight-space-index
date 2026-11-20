import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import moi_loop_router_vllm as lr  # noqa: E402
import rescue_answer as ra  # noqa: E402


class CharTok:
    """Token id = character code, so positions in tokens and characters match."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_explicit_answers_boxed_and_phrases():
    t = r"so the answer is 1,200. Wait. Final answer: $45 and \boxed{7}"
    assert [a for _, a in ra.explicit_answers(t)] == ["1200", "45", "7"]


def test_explicit_answers_ignores_plain_numbers():
    assert ra.explicit_answers("3 + 4 = 7, then 7 * 2 = 14") == []


def test_rescue_strict_only_sees_text_before_onset():
    head = "work... so the answer is 5. hmm "
    loop = "the answer is 9 " * 200
    ids = [ord(c) for c in head + loop]
    kill = len(head) + lr.KILL_WINDOW  # onset lands right after the head
    r = ra.rescue(ids, kill, CharTok())
    assert r["onset"] == len(head)
    assert r["strict"] == "5"
    assert r["medium"] == "9"  # most frequent explicit answer up to the kill
