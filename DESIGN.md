# Toy experiment design (locked 2026-10-04, before any run)

## Setting

- **Model:** Qwen2.5-0.5B-Instruct, bf16, RTX 4090 Laptop (16 GB).
- **Knowledge:** E = 8 fictional entities, each with a fact sheet of 20 attributes (~1–1.5k tokens). Each entity
  links to 2 others ("partner", "competitor"); links are written into the fact sheet.
- **KV segments:** each fact sheet is prefilled once into its own KV segment, in a fixed position slot after a
  shared system prompt. Segments live in a **CPU tier** (pinned memory).
- **GPU budget:** at most `k` segments resident (k ∈ {1, 2, 3}). A follow-up attends to
  system prompt + resident segments + the query.
- **Dialogue traces:** 40 synthetic conversations × 12 follow-ups. The topic follows a Markov chain:
  stay on the entity with p = 0.6, move to a linked entity with p = 0.3, jump to a random entity with p = 0.1.
  Each follow-up asks for one attribute of the current entity ("What is the X of Y? Answer with just the value.").
- **Think time:** assumed long enough for any prefetch issued between turns to finish (prefetch is off the critical
  path). Its cost is still counted in bytes moved and in wasted prefetch.

## Miss cost: two tiers, both reported

- **reload:** copy the segment's KV from CPU to GPU on the critical path.
- **recompute:** re-prefill the segment on the critical path (no CPU tier). This stands in for long or large-model
  contexts, where the miss cost is large.

## Methods (all at the same budget k)

| Name | Before the query (think time) | When the query arrives |
|---|---|---|
| `reactive` | nothing | pick the segment named in the query (EpiCache-style); load it on a miss; LRU eviction |
| `lru-warm` | keep the k most recently used segments | same fallback |
| `topic` | **H1:** warm the active entity + its linked entities (by link order), up to k | same fallback |
| `oracle` | warm exactly the next query's segment | — |
| `all-resident` | no budget; every segment on the GPU | — (reference for the memory cost) |

The `topic` heuristic is a fixed rule (no learned head), as the handoff specifies.

## Metrics (logged per turn)

Follow-up TTFT in ms (median over turns, after warm-up), accuracy (exact containment of the gold value),
prefetch hit rate, wasted prefetch (warmed and evicted without being used), MB moved, and peak GPU memory.

## Success threshold (GO only if all hold, at k = 2, in the recompute tier)

1. `topic` median TTFT is ≥ 15% lower than `reactive`.
2. `topic` beats `lru-warm` on median TTFT (the predicted topic must add something beyond recency).
3. `topic` accuracy is within 1 point of `all-resident`.
4. `topic` hit rate is > 50%.

## Known bias (stated in advance)

I wrote the transition structure *and* the heuristic, and the heuristic reads the same links the trace generator
uses. Toy signal therefore only gives permission to run a real test (a real multi-turn dataset, a learned
predictor, an 8B model). It is not a result. The reload tier on a 0.5B model is expected to show tiny absolute
savings (about 12 KB/token of KV), and that will be reported as is.
