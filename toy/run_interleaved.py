"""Interleaved re-run: every policy steps through the SAME turn back-to-back, in shuffled order.

The first run (run.py) executed configurations one after another. The laptop GPU switches between two clock states
(~20 ms vs ~50 ms query TTFT), which made the between-config latency differences meaningless.
Here, drift hits every method equally, and comparisons are paired per turn.

Usage: .venv/Scripts/python toy/run_interleaved.py
"""
import csv
import json
import random
import statistics as st
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import ATTRIBUTES, ENTITIES, make_traces, make_world  # noqa: E402
from engine import Engine  # noqa: E402
from run import engine_ttft, want_set  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"


class Policy:
    def __init__(self, method, k, links, eng):
        self.method, self.k, self.links, self.eng = method, k, links, eng
        self.reset()

    def reset(self):
        self.resident, self.unused, self.prev = OrderedDict(), set(), None

    def _evict(self, n):
        self.resident.pop(n)
        if n in self.unused:
            self.unused.discard(n)
            return 1
        return 0

    def step(self, ent, attr):
        eng, wasted, pre_mb = self.eng, 0, 0.0
        if self.method == "on-demand":
            self.resident.clear()
        want = want_set(self.method, self.prev, ent, self.links, self.k)
        if want is not None:  # think-time prefetch, off the critical path
            for n in [n for n in self.resident if n not in want]:
                wasted += self._evict(n)
            for n in want:
                if n not in self.resident:
                    self.resident[n] = eng.reload(n)[0]
                    pre_mb += eng.seg_bytes[n] / 1e6
                    self.unused.add(n)
        hit = ent in self.resident
        reload_ms = recompute_ms = 0.0
        if not hit:
            _, reload_ms = eng.reload(ent)
            self.resident[ent], recompute_ms = eng.recompute(ent)
            self.resident.move_to_end(ent)
            while len(self.resident) > max(self.k, 1):
                wasted += self._evict(next(iter(self.resident)))
        self.unused.discard(ent)
        self.resident.move_to_end(ent)
        q_ms = engine_ttft(eng, self.resident[ent], attr, ent)
        self.prev = ent
        return dict(hit=int(hit), reload_ms=reload_ms, recompute_ms=recompute_ms, query_ms=q_ms,
                    ttft_reload=q_ms + reload_ms, ttft_recompute=q_ms + recompute_ms,
                    prefetch_mb=pre_mb, wasted=wasted)

    def end_trace(self):
        w = len(self.unused)
        self.reset()
        return w


def main():
    facts, links, sheets = make_world(0)
    traces = make_traces(links, n_traces=40, n_turns=12)
    eng = Engine()
    eng.build_segments(sheets)
    warm = random.Random(3)
    for _ in range(60):
        e = warm.choice(ENTITIES)
        engine_ttft(eng, eng.reload(e)[0], warm.choice(ATTRIBUTES), e)
        eng.recompute(e)

    configs = [("on-demand", 1)] + [(m, k) for k in (1, 2, 3) for m in ("reactive-lru", "topic", "oracle")] \
        + [("all-resident", len(ENTITIES))]
    pols = {f"{m}-k{k}": Policy(m, k, links, eng) for m, k in configs}
    order_rng = random.Random(11)
    rows, wasted_tail = [], {c: 0 for c in pols}
    for ti, trace in enumerate(traces):
        for t, (ent, attr) in enumerate(trace):
            names = list(pols)
            order_rng.shuffle(names)
            for c in names:
                r = pols[c].step(ent, attr)
                rows.append(dict(cfg=c, trace=ti, turn=t, entity=ent, attr=attr, **r))
        for c, p in pols.items():
            wasted_tail[c] += p.end_trace()
        print(f"trace {ti + 1}/{len(traces)}", flush=True)

    with open(OUT / "toy_turns_interleaved.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    (OUT / "wasted_tail.json").write_text(json.dumps(wasted_tail, indent=1))


if __name__ == "__main__":
    main()
