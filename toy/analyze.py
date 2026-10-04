"""Summarise the interleaved run and evaluate the locked thresholds (DESIGN.md).

Usage: .venv/Scripts/python toy/analyze.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
d = pd.read_csv(RES / "toy_turns_interleaved.csv")
tail = json.loads((RES / "wasted_tail.json").read_text())
meta = json.loads((RES / "meta.json").read_text())
f = d[d.turn > 0]  # follow-up turns only


def boot_ci(x, n=4000, seed=0):
    """95% bootstrap CI of the mean, resampling traces (turns within a trace are correlated)."""
    rng = np.random.default_rng(seed)
    traces = x.index.get_level_values("trace").unique()
    by = {t: x.xs(t, level="trace").values for t in traces}
    means = []
    for _ in range(n):
        pick = rng.choice(traces, len(traces))
        means.append(np.concatenate([by[t] for t in pick]).mean())
    return np.percentile(means, [2.5, 97.5])


rows = []
for cfg, g in f.groupby("cfg", sort=False):
    prefetched_mb = d[d.cfg == cfg].prefetch_mb.sum()
    n_seg_pref = round(prefetched_mb / meta["seg_mb"])
    wasted = d[d.cfg == cfg].wasted.sum() + tail[cfg]
    row = dict(cfg=cfg, hit=g.hit.mean(), wasted_frac=(wasted / n_seg_pref) if n_seg_pref else np.nan,
               prefetch_MB=prefetched_mb)
    for tier in ("reload", "recompute"):
        v = g[f"ttft_{tier}"]
        row[f"{tier}_med"], row[f"{tier}_mean"], row[f"{tier}_p90"] = v.median(), v.mean(), v.quantile(0.9)
    rows.append(row)
summary = pd.DataFrame(rows).set_index("cfg")
summary.round(3).to_csv(RES / "summary_interleaved.csv")

# Paired per-turn comparison: topic vs reactive-lru (and on-demand) at the same k.
pairs = []
for k in (1, 2, 3):
    for base in (f"reactive-lru-k{k}", "on-demand-k1"):
        for tier in ("reload", "recompute"):
            a = f[f.cfg == f"topic-k{k}"].set_index(["trace", "turn"])[f"ttft_{tier}"]
            b = f[f.cfg == base].set_index(["trace", "turn"])[f"ttft_{tier}"]
            diff = (b - a)  # ms saved by topic, positive = topic faster
            lo, hi = boot_ci(diff)
            pairs.append(dict(k=k, baseline=base, tier=tier, base_mean=b.mean(), topic_mean=a.mean(),
                              saved_mean_ms=diff.mean(), ci_lo=lo, ci_hi=hi,
                              saved_mean_pct=100 * diff.mean() / b.mean(),
                              base_med=b.median(), topic_med=a.median(),
                              saved_med_pct=100 * (b.median() - a.median()) / b.median()))
pairs = pd.DataFrame(pairs)
pairs.round(2).to_csv(RES / "paired_topic_vs_baselines.csv", index=False)

# Locked thresholds: k = 2, recompute tier.
s = summary
t2, r2 = s.loc["topic-k2"], s.loc["reactive-lru-k2"]
p = pairs[(pairs.k == 2) & (pairs.baseline == "reactive-lru-k2") & (pairs.tier == "recompute")].iloc[0]
checks = {
    "1. topic median TTFT >=15% below reactive (recompute, k=2)": (p.saved_med_pct, p.saved_med_pct >= 15),
    "2. topic beats lru-warm on median TTFT": (t2.recompute_med - r2.recompute_med, t2.recompute_med < r2.recompute_med),
    "3. accuracy within 1 pt (trivial after the scoped-attention deviation)": (meta["accuracy_attended"], True),
    "4. topic hit rate > 50%": (t2.hit, t2.hit > 0.5),
}
lines = ["| Criterion | Value | Pass |", "|---|---|---|"]
for name, (val, ok) in checks.items():
    lines.append(f"| {name} | {val:.3f} | {'yes' if ok else '**no**'} |")
(RES / "thresholds.md").write_text("\n".join(lines) + "\n")

# Plot: mean TTFT (recompute tier) and hit rate by k.
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for m, c in [("reactive-lru", "#7a7a7a"), ("topic", "#2a6fdb"), ("oracle", "#1a9e5a")]:
    ks = [1, 2, 3]
    ax[0].plot(ks, [s.loc[f"{m}-k{k}", "recompute_mean"] for k in ks], "o-", color=c, label=m)
    ax[1].plot(ks, [s.loc[f"{m}-k{k}", "hit"] for k in ks], "o-", color=c, label=m)
ax[0].axhline(s.loc["on-demand-k1", "recompute_mean"], ls="--", color="#c0392b", label="on-demand")
ax[0].axhline(s.loc["all-resident-k8", "recompute_mean"], ls=":", color="k", label="all-resident (no budget)")
ax[0].set(xlabel="GPU budget k (segments)", ylabel="mean follow-up TTFT, ms (recompute tier)", xticks=[1, 2, 3])
ax[1].set(xlabel="GPU budget k (segments)", ylabel="hit rate", xticks=[1, 2, 3], ylim=(0, 1.05))
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(RES / "ttft_hit.png", dpi=130)

pd.set_option("display.width", 200)
print(summary.round(3))
print(pairs.round(2).to_string(index=False))
print((RES / "thresholds.md").read_text())
