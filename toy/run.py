"""Run the toy: policies x GPU budget, measured TTFT in two miss tiers, plus a cheap hit-rate sweep.

Usage: .venv/Scripts/python toy/run.py
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

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)


def want_set(method, prev, nxt, links, k):
    """What to hold on the GPU after think-time prefetch; None means keep the current set (no prefetch)."""
    if method == "topic":
        return ([prev] + links[prev])[:k] if prev else None
    if method == "oracle":
        return [nxt]
    if method == "all-resident":
        return list(ENTITIES)
    return None


def simulate(method, k, traces, links, engine=None, facts=None):
    """Replay traces through one policy. With an engine, every miss is measured in both tiers."""
    rows = []
    for ti, trace in enumerate(traces):
        resident = OrderedDict()  # name -> GPU KV (or True when simulating only); LRU order
        unused_prefetch = set()
        prev = None
        for t, (ent, attr) in enumerate(trace):
            prefetch_bytes, wasted = 0, 0
            if method == "on-demand":
                resident.clear()
            want = want_set(method, prev, ent, links, k)
            if want is not None:
                for n in [n for n in resident if n not in want]:
                    resident.pop(n)
                    if n in unused_prefetch:
                        wasted += 1
                        unused_prefetch.discard(n)
                for n in want:
                    if n not in resident:
                        resident[n] = engine.reload(n)[0] if engine else True  # off the critical path
                        prefetch_bytes += engine.seg_bytes[n] if engine else 0
                        unused_prefetch.add(n)
            hit = ent in resident
            reload_ms = recompute_ms = 0.0
            if not hit:
                if engine:
                    _, reload_ms = engine.reload(ent)
                    resident[ent], recompute_ms = engine.recompute(ent)
                else:
                    resident[ent] = True
                while len(resident) > max(k, 1):
                    n, _ = resident.popitem(last=False)
                    if n in unused_prefetch:
                        wasted += 1
                        unused_prefetch.discard(n)
            unused_prefetch.discard(ent)
            resident.move_to_end(ent)
            row = dict(method=method, k=k, trace=ti, turn=t, entity=ent, attr=attr, hit=int(hit),
                       prefetch_mb=prefetch_bytes / 1e6, wasted=wasted, resident=len(resident))
            if engine:
                q_ms = engine_ttft(engine, resident[ent], attr, ent)
                row.update(reload_ms=reload_ms, recompute_ms=recompute_ms, query_ms=q_ms,
                           ttft_reload=q_ms + reload_ms, ttft_recompute=q_ms + recompute_ms)
            rows.append(row)
            prev = ent
        # prefetches still unused at the end of a trace count as wasted
        if rows:
            rows[-1]["wasted"] += len(unused_prefetch)
    return rows


def engine_ttft(engine, seg, attr, ent):
    """Query prefill + first token over the attended segment (no decoding needed for latency)."""
    import torch
    from engine import QUERY_TMPL, _sync
    import time
    cache = engine._assemble({ent: seg})
    ids = engine._ids(QUERY_TMPL.format(q=f"What is the {attr} of {ent}?"))
    pos = torch.arange(ids.shape[1], device=engine.device).unsqueeze(0) + engine.query_start
    with torch.no_grad():
        _sync()
        t0 = time.perf_counter()
        out = engine.model(ids, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
        out.logits[:, -1].argmax(-1)
        _sync()
    return (time.perf_counter() - t0) * 1e3


def summarize(rows, key_cols=("method", "k")):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[c] for c in key_cols), []).append(r)
    out = []
    for key, rs in groups.items():
        follow = [r for r in rs if r["turn"] > 0]
        pre_mb = sum(r["prefetch_mb"] for r in rs)
        n_pref = sum(1 for r in rs if r["prefetch_mb"] > 0)
        s = dict(zip(key_cols, key), hit_rate=st.mean(r["hit"] for r in follow),
                 wasted=sum(r["wasted"] for r in rs), prefetch_mb_total=round(pre_mb, 1), prefetch_events=n_pref)
        if "ttft_reload" in rs[0]:
            for tier in ("reload", "recompute"):
                v = [r[f"ttft_{tier}"] for r in follow]
                s[f"ttft_{tier}_median"] = round(st.median(v), 2)
                s[f"ttft_{tier}_mean"] = round(st.mean(v), 2)
                s[f"ttft_{tier}_p90"] = round(sorted(v)[int(0.9 * len(v))], 2)
        out.append(s)
    return out


def accuracy_check(engine, facts, traces):
    """Accuracy depends only on the attended segment, which is identical across methods. Verify once."""
    ok, n = 0, 0
    for trace in traces:
        for ent, attr in trace:
            seg, _ = engine.reload(ent)
            _, txt = engine.answer({ent: seg}, f"What is the {attr} of {ent}?", max_new=12)
            ok += facts[ent][attr].lower() in txt.lower()
            n += 1
    return ok / n, n


def main():
    facts, links, sheets = make_world(0)
    traces = make_traces(links, n_traces=40, n_turns=12)
    (OUT / "world.json").write_text(json.dumps({"links": links, "facts": facts}, indent=1))

    # 1) Cheap hit-rate sweep over the transition structure (no model): the falsification check.
    sweep = []
    for p_stay, p_link in [(0.6, 0.3), (0.6, 0.0), (0.3, 0.6), (0.9, 0.05), (0.2, 0.2)]:
        tr = make_traces(links, n_traces=400, n_turns=12, p_stay=p_stay, p_link=p_link, seed=7)
        for k in (1, 2, 3):
            for m in ("on-demand", "reactive-lru", "topic", "oracle"):
                rs = simulate(m, k, tr, links)
                sweep.append(dict(p_stay=p_stay, p_link=p_link, method=m, k=k,
                                  hit_rate=round(st.mean(r["hit"] for r in rs if r["turn"] > 0), 3),
                                  wasted_per_turn=round(sum(r["wasted"] for r in rs) / len(rs), 3)))
    with open(OUT / "hit_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sweep[0].keys())
        w.writeheader()
        w.writerows(sweep)
    print("hit-rate sweep written")

    # 2) Measured run on the GPU.
    eng = Engine()
    eng.build_segments(sheets)
    print(f"segment ~{st.mean(eng.seg_tokens.values()):.0f} tok, {st.mean(eng.seg_bytes.values())/1e6:.1f} MB")
    warm = random.Random(3)
    for _ in range(30):  # warm-up
        e = warm.choice(ENTITIES)
        engine_ttft(eng, eng.reload(e)[0], warm.choice(ATTRIBUTES), e)
        eng.recompute(e)

    acc, n = accuracy_check(eng, facts, traces)
    print(f"accuracy (attended segment only): {acc:.3f} over {n}")

    rows = []
    configs = [("on-demand", 1)] + [(m, k) for k in (1, 2, 3) for m in ("reactive-lru", "topic", "oracle")] \
        + [("all-resident", len(ENTITIES))]
    for m, k in configs:
        rows += simulate(m, k, traces, links, engine=eng)
        print("done", m, k)
    with open(OUT / "toy_turns.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows)
    with open(OUT / "toy_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summ[0].keys())
        w.writeheader()
        w.writerows(summ)
    (OUT / "meta.json").write_text(json.dumps(dict(
        accuracy_attended=acc, accuracy_n=n, seg_tokens=st.mean(eng.seg_tokens.values()),
        seg_mb=st.mean(eng.seg_bytes.values()) / 1e6), indent=1))
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
