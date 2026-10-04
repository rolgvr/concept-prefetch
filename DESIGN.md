# H2: environment-cued graded activation (locked 2026-10-04, before any data is read)

Origin: the observation of "drifting thoughts": associations that pop up almost instantly from cues in the
environment, most of which fade unused. Reading for systems: cheap, broad, graded activation, with only a
few candidates promoted to expensive warm state.

**H2:** Across real multi-turn sessions, a predictor that combines conversation cues *and* environment cues into
a decaying activation score anticipates the context the next query needs better than recency, and better than
conversation-only prediction.

**H2-null:** Recency matches it within 10 points (conversations are local), or environment cues add < 5 points over
conversation-only.

This is a **data / predictability** test (no GPU). Latency follows from hit rate × miss cost, as the toy showed.

## Datasets
- **A. Local coding sessions** (my own coding-assistant session logs; raw data never leaves the machine and only
  aggregate metrics are committed). Segment = a file. Needed set for turn t+1 = files the assistant opens or edits
  while answering user turn t+1. Environment cues = file paths appearing in tool *outputs* (tracebacks, grep/ls
  results, test failures) and files touched. Conversation cues = paths or file names mentioned in the user or
  assistant *text*.
- **B. TopiOCQA** (public, topic-switching conversational QA). Segment = a topic page. Cue = the next topic's title
  appearing in the current turn's text. Tests H1's "predictable topic drift" on public data.

## Predictors (all ranked lists; the warm set = top-k)
1. `recency`: the k most recently needed segments.
2. `conversation`: decaying activation from conversation cues only.
3. `environment`: decaying activation from environment cues only.
4. `graded` (H2): activation = Σ cue weights × decay^age, from both sources plus recency.
   - Tiers by rank: top k = GPU; ranks k+1..4k = CPU staging (shallow).
   - Reported: hit@k (GPU), hit@4k (shallow), wasted fraction.

Weights and decay are fit on a **dev split** (the first 30% of sessions by time), then frozen and evaluated once
on the remaining 70%.

## Metrics
- **hit@k:** share of needed segments at turn t+1 that were in the warm set built before the query.
  Only segments seen before (in any cue) count as reachable. Also reported: the reachable ceiling, and hit
  over all segments.
- **Wasted:** share of warmed segments not used at turn t+1.

## Success threshold (test split, k = 4)
1. `graded` hit@k ≥ `recency` hit@k + 10 points.
2. `graded` hit@k ≥ `conversation` hit@k + 5 points (the environment contributes).
3. Both hold on dataset A; dataset B is reported as a secondary result.

---

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
