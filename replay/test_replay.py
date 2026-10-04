"""Sanity tests on synthetic sessions with a known answer. Run: .venv/Scripts/python replay/test_replay.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay import evaluate, run  # noqa: E402


def synth(env_leads, n=60, turns=15, seed=0):
    """If env_leads, the segment needed at t+1 appears in turn t's environment cues (an error trace)."""
    rng = random.Random(seed)
    files = [f"f{i}.py" for i in range(30)]
    sessions = []
    for i in range(n):
        nxt = [rng.choice(files) for _ in range(turns + 1)]
        ts = []
        for t in range(turns):
            env = [nxt[t + 1]] + rng.sample(files, 3) if env_leads else rng.sample(files, 4)
            ts.append({"conv": rng.sample(files, 2), "env": env, "needed": [nxt[t]]})
        sessions.append({"t0": i, "turns": ts})
    return sessions


def test_env_signal_detected():
    res = run(synth(True), k=4)
    m = res["methods"]
    assert m["graded"]["test"]["hit_at_k"] > 0.9, m["graded"]
    assert m["environment"]["test"]["hit_at_k"] > 0.9
    assert m["recency"]["test"]["hit_at_k"] < 0.3
    assert res["thresholds"]["graded >= recency + 10pt"][1]


def test_no_signal_no_false_positive():
    res = run(synth(False), k=4)
    m = res["methods"]
    assert m["graded"]["test"]["hit_at_k"] < 0.35, m["graded"]
    assert not res["thresholds"]["graded >= recency + 10pt"][1]


def test_first_turn_not_scored():
    s = [{"t0": 0, "turns": [{"conv": [], "env": [], "needed": ["a"]}]}]
    assert evaluate(s, "recency", {}, 4)["n_needed"] == 0


if __name__ == "__main__":
    for f in (test_env_signal_detected, test_no_signal_no_false_positive, test_first_turn_not_scored):
        f()
        print("ok", f.__name__)
