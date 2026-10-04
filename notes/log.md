# Lab notebook

Running notes, in order. Kept as written at the time, including the dead ends.

## 2026-10-04

**Setup.** I changed the toy baseline from plain prefix caching to on-demand reload. With prefix caching the topic's
KV is already on the GPU, so a follow-up only prefills its own ~20 tokens and prefetch has nothing to save. Prefetch
can only matter when the context lives in a slower tier (offloaded, evicted, compressed, or retrieved) between turns.

**Literature check.** About 45 papers. Verdict: partially covered (see `novelty.md`). Closest work: VoiceAgentRAG
(topic-predicted prefetch of retrieved text), EpiCache (topic-segmented KV, selected after the query arrives), and
PerCache (predicted queries to precomputed KV, across sessions).

**Design locked** in `DESIGN.md` and committed before any run. Environment: Python 3.12, torch 2.14 + cu126,
transformers 5.18.

**Sanity check, which forced a design change.** Qwen2.5-0.5B, fact-sheet KV segments prefilled independently at
fixed RoPE slots (~776 tokens, 9.5 MB each):

| Attention scope | Accuracy (n=40) |
|---|---|
| queried segment only | 97.5% |
| queried + 1 other | 42.5% |
| queried + 2 others | 35% |
| all 8 segments | 12.5% |
| full context, normal prefill (6.2k tokens) | 67.5% |

Independently encoded segments interfere when attended together (the cross-segment issue CacheBlend describes).
I separated GPU residency from attention scope: each follow-up attends only to the segment it names, as EpiCache does.
Because of that, accuracy is the same across methods, and the toy only measures latency.

**Miss costs.** Reload from pinned CPU memory ~0.8 ms, recompute ~79 ms, query TTFT ~50 ms. At this model size the
reload tier cannot show a 15% gain.

**Run 1 discarded.** Configurations ran one after another, and the laptop GPU moved between two clock states
(~20 vs ~50 ms). Two policies with identical behaviour measured 22.5 vs 49.9 ms. I re-ran with all policies
interleaved on each turn and compared paired.

**Toy result.** Recompute tier: at k=2, topic prefetch beats reactive LRU by 6.2% in the mean (95% CI 2.6–10%), and by
19.4% at k=3. The pre-registered median criterion passes (23.6%), but the median sits between the two clock modes and
is not reliable. Reload tier: no gain. Wasted prefetch 60–68%. With no topic links in the traces, the advantage
vanishes (hit rate 0.71 vs 0.70).

**Second hypothesis (H2): environment-cued graded activation.** Locked in `DESIGN.md` before any data was read.
Literature: each piece exists (spreading-activation memory, proactive assistants driven by environment events,
confidence-tiered KV), but I found no work that combines them.

**H2 on my own coding-assistant session logs** (parsed locally, aggregates only). 5 usable sessions, 1,018 turns.
Test split (4 sessions, 1,595 file needs), hit@4:

| recency | conversation | environment | graded |
|---|---|---|---|
| 10.5% | 0.3% | 6.1% | 11.0% |

Criterion 1 fails (+0.5 pt vs +10 required). The dev split ended up as a single session with 10 events, so the fit
was degenerate. I should have checked the session count before choosing a 30% split.

**H2 upper bound** (tuned in-sample on all sessions, deliberately optimistic):

| k | recency | graded | gain |
|---|---|---|---|
| 2 | 7.2% | 10.0% | +2.7 pt |
| 4 | 10.5% | 12.3% | +1.9 pt |
| 8 | 15.5% | 17.4% | +1.9 pt |

No file seen in the environment cues was later needed without having already been touched (0.0%), and 56% of needed
files were new to the session. In agentic coding the agent acts on the environment within the same turn, so there
is little left to predict across the gap between turns. The null holds for this setting. It says nothing yet about
settings where a person, not an agent, drifts between topics.
