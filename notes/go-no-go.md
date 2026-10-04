# Go / no-go: day 1 (2026-10-04)

**Verdict: conditional GO.** The idea is partially covered but has a defensible niche. The toy passes the locked
thresholds as written, but the robust effect is smaller than the headline. The toy cannot answer the decisive
question: do real conversations move between topics in a way a topic predictor can anticipate?

## Novelty

Partially covered ([novelty.md](novelty.md)). Closest work: VoiceAgentRAG (topic-predicted prefetch of *text*
between turns), EpiCache (topic-segmented KV, selected *after* the query arrives), PerCache (predicted queries →
KV, cross-session). Open niche: topic-predicted promotion of conversation KV segments into a bounded GPU budget
during think time.

## Toy result

Qwen2.5-0.5B, RTX 4090 Laptop, 8 entities × 1 KV segment (~770 tokens, 9.4 MB), 40 traces × 12 turns, all policies
interleaved per turn (paired). Accuracy with scoped attention: 95.0% (n = 480), identical across methods by
construction.

Recompute tier (a miss re-prefills the segment, ~25–60 ms), mean follow-up TTFT:

| k | on-demand | reactive-LRU | **topic** | oracle | topic saving vs LRU (paired mean, 95% CI) | topic hit rate | wasted prefetch |
|---|---|---|---|---|---|---|---|
| 1 | 72.0 | 48.8 | 49.6 | 33.9 | −1.7% (n.s.) | 60% | — |
| 2 | 72.0 | 46.7 | **43.8** | 33.1 | **6.2%** (2.6–10%) | 74% | 68% |
| 3 | 72.0 | 44.5 | **35.8** | 33.6 | **19.4%** (14–25%) | 93% | 60% |

all-resident (no budget, 8× the memory): 33.0 ms. Reload tier (miss = 0.5 ms PCIe copy): no difference
between any methods (±1 ms).

### Locked thresholds (k = 2, recompute)

| Criterion | Value | Pass |
|---|---|---|
| Median TTFT ≥ 15% below reactive | 23.6% | yes, but **fragile**, see below |
| Beats LRU on median | −9.9 ms | yes |
| Accuracy within 1 point | 95.0% | yes (trivial after the scoped-attention deviation) |
| Hit rate > 50% | 74% | yes |

**Why the median pass is weak:** the laptop GPU switches between two clock states, so per-turn TTFT is bimodal
(~20 ms / ~45 ms). The median falls in the gap and jumps with small shifts in composition. The paired mean is 6.2% at
k = 2, which is **below** the 15% bar. Only k = 3 clears 15% on the mean.

## What the toy actually shows

1. **The gain is entirely hit rate × miss cost.** With a cheap miss (a small model, KV in pinned CPU memory), the
   gain is zero. The mechanism only matters when a miss costs as much as a prefill: large models, long contexts,
   disk or network tiers, or no CPU tier.
2. **The advantage comes only from predictable topic structure.** Hit-rate sweep at k = 2 (400 traces):

   | Conversation dynamics | LRU hit | topic hit |
   |---|---|---|
   | stays 60%, follows links 30% | 0.68 | 0.79 |
   | stays 60%, **never follows links** | 0.70 | 0.71 |
   | stays 30%, follows links 60% | 0.45 | 0.64 |
   | stays 90% | 0.91 | 0.94 |
   | mostly random jumps | 0.37 | 0.46 |

   I wrote the links into both the trace generator and the heuristic, so this is toy bias by design.
3. **Wasted prefetch is high (60–68%)**, consistent with the bandwidth objection in 2609.16215. Topic prefetch moved
   1.7 GB across 480 turns at k = 2. This is cheap here, but it would not be under multi-tenant PCIe contention.
4. **Side result:** independently encoded KV segments interfere when attended together (97.5% → 42.5% with 2
   segments). Scoped attention beat contiguous full context on this model (95–97.5% vs 67.5%).

## Recommended next experiment (only if continuing)

The crux is a **data question, not a systems question**: are real topic transitions predictable from the active
topic? That can be answered cheaply, without a GPU, by replaying a real topic-shifting conversational dataset
through the same hit-rate simulator. **TopiOCQA** (conversational QA over Wikipedia whose topic switches follow
hyperlinks; not yet opened in this session, so verify first) is the obvious candidate. The "links" become
Wikipedia hyperlinks, and the predictor sees only the current page.

- If topic-hit beats LRU-hit by ≥ 10 points on real data → build the systems experiment on an 8B model with a real
  miss cost and a learned predictor, against EpiCache and VoiceAgentRAG-style baselines.
- If not → stop. Then the honest result is that "conversations are mostly local; recency already captures it".
