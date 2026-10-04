# Concept-Level Speculative Prefetch

**Question:** in multi-turn dialogue, does pre-warming the representations an *active topic* predicts you'll need
(during the user's think time) cut follow-up latency compared with on-demand loading, without hurting accuracy?

Hypothesis by Rolando Gavrila, drawn from introspection on aphantasia: a persistent latent, query-first addressing,
and eager topic-driven prefetch. Claude is used as the research and coding tool.

Status: **day 1, novelty check in progress.** A null result is a valid outcome and will be reported as one.

## Hypotheses

- **H1:** Pre-warming representations for semantically anticipated sub-queries of the active topic reduces
  follow-up latency against a reactive baseline, with no accuracy loss.
- **H0:** No latency gain, or a gain cancelled out by accuracy loss, wasted prefetch, or memory overhead.

## Design note: why the baseline matters

With plain prefix caching, the topic's KV entries are already resident on the GPU. A follow-up then only prefills
its own ~20 tokens, so prefetch has nothing to save, and a toy run under that baseline would be null by
construction. Prefetch can only pay off when there is a **memory hierarchy**, meaning the context is offloaded,
evicted, compressed, or retrieved between turns. In that setting the topic predictor decides which segments to
load or prefill during the user's idle time. The toy baseline is therefore *on-demand reload*, not *prefix caching*.

## Metrics (locked before any run)

| Metric | Definition |
|---|---|
| Follow-up TTFT | time to first token on follow-up turns, ms (median over N runs, after warm-up) |
| Accuracy | task score, same scorer for both arms |
| Prefetch hit rate | share of follow-ups whose needed segment was already warm |
| Wasted prefetch | share of warmed segments never used |
| Memory overhead | peak extra memory vs baseline, MB |

**Success threshold (draft, to be locked in step 2):** ≥15% lower median follow-up TTFT, accuracy within 1 point,
hit rate > 50%.

## Plan

1. Novelty check (arXiv, Semantic Scholar, ICML/NeurIPS/ICLR/ACL 2025–26). Verdict: novel / partially covered / taken.
2. Lock the design: model, task, baseline, threshold.
3. Toy experiment: small open model, synthetic entity with 20 attributes, 5–10 follow-ups.
4. Go/no-go note.

## Log

- **2026-10-04:** Repo created. Changed the toy baseline from prefix caching to on-demand reload (see design note).
  Novelty check started.
- **2026-10-04:** Novelty check done, ~45 papers opened ([notes/novelty.md](notes/novelty.md)).
  Verdict: **partially covered.** Closest prior work: VoiceAgentRAG (topic-predicted text prefetch), EpiCache
  (topic-segmented KV, selected after the query arrives), PerCache (predicted queries → KV, cross-session).
  The open niche is narrow: topic-predicted promotion of conversation KV segments into a bounded GPU budget during
  think time. Required baselines: EpiCache-reactive, reload-all, on-demand, oracle.
