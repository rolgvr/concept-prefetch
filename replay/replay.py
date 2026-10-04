"""Dataset-agnostic replay for H2: can cues available *before* a query predict the segments it will need?

Input: normalized sessions (JSON list). Each session is {"t0": sortable start time, "turns": [turn, ...]}, and each turn is
    {"conv": [seg, ...],    # segments mentioned in user/assistant text during this turn  (conversation cues)
     "env":  [seg, ...],    # segments appearing in tool outputs / environment            (environment cues)
     "needed": [seg, ...]}  # segments actually opened/used while answering this turn
Before turn t+1's query arrives, a predictor sees turns 0..t and outputs a ranked list; the warm set is the top-k.

Usage: .venv/Scripts/python replay/replay.py <sessions.json> [--k 4] [--out results/h2_<name>.json]
"""
import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


class Activation:
    """Decaying activation per segment. weights: dict source -> weight; sources: conv, env, touch."""

    def __init__(self, weights, decay):
        self.w, self.decay, self.a = weights, decay, defaultdict(float)

    def update(self, turn):
        for s in self.a:
            self.a[s] *= self.decay
        for src, key in (("conv", "conv"), ("env", "env"), ("touch", "needed")):
            w = self.w.get(src, 0.0)
            if w:
                for s in set(turn[key]):
                    self.a[s] += w

    def ranked(self):
        return [s for s, v in sorted(self.a.items(), key=lambda kv: -kv[1]) if v > 0]


class Recency:
    def __init__(self):
        self.order = []

    def update(self, turn):
        for s in turn["needed"]:
            if s in self.order:
                self.order.remove(s)
            self.order.insert(0, s)

    def ranked(self):
        return list(self.order)


def make(name, params):
    if name == "recency":
        return Recency()
    w = {"conversation": {"conv": 1.0}, "environment": {"env": 1.0}}.get(name)
    if w is not None:
        return Activation(w, params["decay"])
    return Activation({"conv": params["w_conv"], "env": params["w_env"], "touch": params["w_touch"]}, params["decay"])


def evaluate(sessions, name, params, k):
    hit = hit_shallow = need = reach = warm_n = wasted = 0
    for sess in sessions:
        pred, seen = make(name, params), set()
        for t, turn in enumerate(sess["turns"]):
            needed = set(turn["needed"])
            if t > 0 and needed:
                ranked = pred.ranked()
                gpu, cpu = set(ranked[:k]), set(ranked[: 4 * k])
                hit += len(needed & gpu)
                hit_shallow += len(needed & cpu)
                need += len(needed)
                reach += len(needed & seen)
                warm_n += len(gpu)
                wasted += len(gpu - needed)
            pred.update(turn)
            seen |= set(turn["conv"]) | set(turn["env"]) | needed
    return dict(hit_at_k=hit / need if need else math.nan, hit_at_4k=hit_shallow / need if need else math.nan,
                hit_given_reachable=hit / reach if reach else math.nan,
                reachable_ceiling=reach / need if need else math.nan,
                wasted_frac=wasted / warm_n if warm_n else math.nan, n_needed=need)


GRID_DECAY = [0.3, 0.5, 0.7, 0.9]
GRID_W = [0.0, 0.5, 1.0, 2.0]


def fit(dev, name, k):
    """Pick hyper-parameters on the dev split only."""
    if name == "recency":
        return {}
    if name in ("conversation", "environment"):
        cands = [{"decay": d} for d in GRID_DECAY]
    else:
        cands = [dict(decay=d, w_conv=a, w_env=b, w_touch=c)
                 for d, a, b, c in itertools.product(GRID_DECAY, GRID_W, GRID_W, GRID_W) if a + b + c > 0]
    return max(cands, key=lambda p: evaluate(dev, name, p, k)["hit_at_k"])


def run(sessions, k=4, dev_frac=0.3):
    sessions = sorted(sessions, key=lambda s: s["t0"])
    n_dev = max(1, int(len(sessions) * dev_frac))
    dev, test = sessions[:n_dev], sessions[n_dev:]
    out = {"k": k, "n_sessions": len(sessions), "n_dev": len(dev), "n_test": len(test),
           "n_turns_test": sum(len(s["turns"]) for s in test), "methods": {}}
    for name in ("recency", "conversation", "environment", "graded"):
        p = fit(dev, name, k)
        out["methods"][name] = {"params": p, "dev": evaluate(dev, name, p, k), "test": evaluate(test, name, p, k)}
    m = out["methods"]
    g, r, c = (m[x]["test"]["hit_at_k"] for x in ("graded", "recency", "conversation"))
    out["thresholds"] = {
        "graded >= recency + 10pt": [round(100 * (g - r), 1), g - r >= 0.10],
        "graded >= conversation + 5pt": [round(100 * (g - c), 1), g - c >= 0.05],
    }
    return out


def coverage(sessions):
    """Where do the needed segments of turn t+1 come from? Share found in turn t's cues (aggregate only)."""
    c = defaultdict(int)
    for sess in sessions:
        ts, seen = sess["turns"], set()
        for t in range(1, len(ts)):
            prev = ts[t - 1]
            seen |= set(prev["conv"]) | set(prev["env"]) | set(prev["needed"])
            for s in set(ts[t]["needed"]):
                c["needed"] += 1
                c["in_prev_env"] += s in set(prev["env"])
                c["in_prev_conv"] += s in set(prev["conv"])
                c["in_prev_needed"] += s in set(prev["needed"])
                c["in_any_earlier_cue"] += s in seen
                c["only_in_env_earlier"] += (s in set(prev["env"])) and s not in set(prev["needed"])
    n = c.pop("needed") or 1
    return {k: round(v / n, 3) for k, v in c.items()} | {"n_needed": n}


def upper_bound(sessions, k):
    """OPTIMISTIC, IN-SAMPLE: tune each method on all sessions. Not a valid estimate, only a ceiling check;
    if graded cannot beat recency by 10 points even here, the null is robust."""
    out = {}
    for name in ("recency", "conversation", "environment", "graded"):
        p = fit(sessions, name, k)
        out[name] = {"params": p, **evaluate(sessions, name, p, k)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--upper-bound", action="store_true", help="in-sample tuning ceiling + cue coverage, k=2,4,8")
    a = ap.parse_args()
    sessions = json.loads(Path(a.sessions).read_text(encoding="utf-8"))
    if a.upper_bound:
        res = {"WARNING": "in-sample tuning; optimistic ceiling, not an estimate",
               "n_sessions": len(sessions), "coverage": coverage(sessions),
               "by_k": {k: upper_bound(sessions, k) for k in (2, 4, 8)}}
        print(json.dumps(res, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        return
    res = run(sessions, k=a.k)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
