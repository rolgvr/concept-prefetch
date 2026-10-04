# Novelty check (2026-10-04)

Closeness: 0 = unrelated … 3 = already does topic-predicted pre-warming for anticipated follow-ups.
Only papers that were actually opened are listed; unverified ones are marked as such.

## Slice B: retrieval-side anticipation

| Paper | What it does | Gap vs H1 | Close |
|---|---|---|---|
| [VoiceAgentRAG](https://arxiv.org/abs/2603.02206) (Salesforce, Mar 2026) | Background LLM predicts 3–5 follow-up topics from the live conversation; pre-retrieves chunks into a FAISS cache between turns. 75% hit rate; retrieval 110 → 0.35 ms on hits | Caches retrieved chunks only; no KV / model-side pre-warming | **3** (retrieval half) |
| [PerCache](https://arxiv.org/abs/2601.11553) (Dec 2025) | LLM predicts future queries in device idle time; precomputes retrieval + chunk QKV; up to 34.4% lower latency | Cross-session long-term interests, not the active topic during think time | **2.5** (KV half) |
| [OnePred](https://arxiv.org/abs/2605.23668) (May 2026) | Next-query prediction from a recursive intent memory, motivated by prefetch | Prediction only, no caching | 1.5 |
| [InstCache](https://arxiv.org/abs/2411.13820) | Offline-predicted instructions → response cache | Global, not conversation-conditioned; responses, not KV | 1.5 |
| [TeleRAG](https://arxiv.org/abs/2502.20969) (MLSys 2026) | Lookahead loading of IVF clusters to GPU | Within one request | 1 |
| [Predictive Prefetching for RAG](https://arxiv.org/abs/2605.17989) (ICML 2026) | Predicts retrieval needs during decoding | Within one generation | 1 |
| [PCR](https://arxiv.org/abs/2603.23049) (Mar 2026) | SSD → DRAM chunk-KV prefetch for queued requests | Reactive | 1 |
| [Yang et al. 2017](https://arxiv.org/abs/1707.05409) | Next-question prediction in conversation | No caching | 1 |
| [IdleSpec](https://arxiv.org/abs/2605.22154) (May 2026) | Speculative planning during tool-wait idle time | Agent idle time, plans not KV | 0.5 |
| ContextCache, RAGCache, Cache-Craft | Reactive semantic / KV caches | No prediction | 0.5 |
| Speculative RAG, PipeRAG, GPT Semantic Cache | — | No future-query prediction | 0 |

Unverified (paywalled): Lempel & Moran WWW 2003; Fagni et al. TOIS 2006 (classical IR result prefetching).

**Slice verdict:** partially covered. Topic-driven between-turn pre-retrieval exists (VoiceAgentRAG), and
KV precompute for predicted queries exists (PerCache, cross-session). The combination of active-topic
prediction during think time with KV pre-prefill, evaluated on TTFT, hit rate and wasted prefill, was not
found in this slice.

## Slice A: serving / KV offload / prefetch

| Paper | What it does | Gap vs H1 | Close |
|---|---|---|---|
| [EpiCache](https://arxiv.org/abs/2509.17396) (Apple, v4 May 2026) | Clusters conversation history into topic "episodes" with a compressed KV per episode; picks the episode by query embedding *after the query arrives*. Up to 2.4× lower latency, 3.7× lower memory | Selection is reactive. Predicting the episode ahead of the query is exactly H1's increment | **2** |
| [PerCache](https://arxiv.org/abs/2601.11553) | (see slice B) predicted queries → precomputed chunk QKV | Single-turn mobile RAG; idle time measured in hours | 2–3 |
| [VoiceAgentRAG](https://arxiv.org/abs/2603.02206) | (see slice B) | Text chunks, not KV | 2 |
| [PBKV](https://arxiv.org/abs/2605.06472) (May 2026) | Predicts upcoming agent invocations; KV eviction and prefetch | Predicts the agent workflow, not the user topic | 2 |
| [CacheScout](https://arxiv.org/abs/2608.14624) (Aug 2026) | Learns agent transitions online; proactive KV prefetch | Same as PBKV | 2 |
| [Don't Wait to Reply](https://arxiv.org/abs/2607.03093) (Jul 2026) | Reasons ahead about likely user replies during pauses; reuses on match | Precomputes reasoning content, not KV | 1–2 |
| [llm-d-router #2949](https://github.com/llm-d/llm-d-router/issues/2949) (Sep 2026, proposal) | After turn N, async prefill of the assistant's answer before turn N+1. Up to 3× TTFT | Uses think time but prefills only known content; no prediction | 1–2 |
| [Where Should the KV Cache Live?](https://arxiv.org/abs/2609.16215) (Sep 2026) | GPU/CPU/SSD placement for long sessions; EWMA next-touch prefetch | **Negative result:** prefetch did not pay for its bandwidth, and even an oracle did not beat no-prefetch on migration traffic. H1 must answer this | 1 |
| [CachedAttention](https://arxiv.org/abs/2403.19708) (ATC 2024) | GPU → host → disk multi-turn KV; preload hinted by the scheduler queue | Reactive to queued requests | 1 |
| [Pensieve](https://arxiv.org/abs/2312.05516) (EuroSys 2025) | Stateful multi-turn GPU/CPU KV tier | Reloads on arrival | 1 |
| [HyMCache](https://arxiv.org/abs/2607.18141) (Jul 2026) | CXL rack; prefetches the whole known prefix | Whole conversation, not a topic subset | 1 |
| [Continuum](https://arxiv.org/abs/2511.02230) (Nov 2025) | Pins KV for the predicted duration of a tool call | Predicts timing, not content | 1 |
| [IntentKV](https://arxiv.org/abs/2606.09916) (Jun 2026) | Cross-turn intent memory used for KV pruning | Looks back and prunes; no prefetch | 1 |
| [PCR](https://arxiv.org/abs/2603.23049) | (see slice B) | Reactive | 1 |
| [SpeCache](https://arxiv.org/abs/2503.16163), [InfiniGen](https://arxiv.org/abs/2406.19707) | Prefetch at the next-token / attention level | The level H1 explicitly excludes | 0–1 |
| [LMCache](https://arxiv.org/abs/2510.09665), [CacheGen](https://arxiv.org/abs/2310.07240), [CacheBlend](https://arxiv.org/abs/2405.16444), [Mooncake](https://arxiv.org/abs/2407.00079), [KVCache in the Wild](https://arxiv.org/abs/2506.02634), [SwiftCache](https://arxiv.org/abs/2606.16135) | Infrastructure for KV storage, transport, reuse, or eviction | No content prediction (useful building blocks) | 0–1 |

Unverified: DualDecoder 2607.26475, CXL-SpecKV 2512.11920, HCache, CacheWise 2606.16824, TurboRAG.

**Slice verdict:** partially covered. No system found promotes conversation KV segments chosen by a *predicted topic*
during inter-turn think time. The pieces exist separately: topic-segmented KV (EpiCache), topic-predicted prefetch of
text or single-turn QKV (VoiceAgentRAG, PerCache), lookahead KV prefetch keyed on agent workflow (PBKV, CacheScout),
and think-time warmup of known content (llm-d).
## Slice C: agent memory / idle-time compute / speculative agents

| Paper | What it does | Gap vs H1 | Close |
|---|---|---|---|
| [ProAct](https://arxiv.org/abs/2605.25971) (May 2026) | "Future-State Prediction" of follow-ups and adjacent topics; in idle time, gathers evidence into memory artifacts | Text artifacts; aims at quality, not latency | 2 |
| [Sleep-time Compute](https://arxiv.org/abs/2504.13171) (Letta/Berkeley 2025) | Re-represents the context offline as inferred text; 5× less test-time compute | No query/topic prediction; text, not KV | 1–2 |
| [Cartridges](https://arxiv.org/abs/2506.06266) (2025) | Trains a compact KV per corpus offline using synthetic self-study | Offline and per corpus; not dialogue-triggered | 1–2 |
| [EpiCache](https://arxiv.org/abs/2509.17396), [VoiceAgentRAG](https://arxiv.org/abs/2603.02206), [CacheScout](https://arxiv.org/abs/2608.14624), [OnePred](https://arxiv.org/abs/2605.23668), [IdleSpec](https://arxiv.org/abs/2605.22154), [IntentKV](https://arxiv.org/abs/2606.09916) | see slices A/B | | |
| [TurboRAG](https://arxiv.org/abs/2410.07590) (EMNLP 2025) | Precomputed KV for every chunk, offline | No prediction (substrate) | 1 |
| [Speculative Interaction Agents](https://arxiv.org/abs/2605.13360) (May 2026) | Speculative tool calls during user think time; 1.3–2.2× | Speculates actions | 1 |
| [Mori et al.](https://arxiv.org/abs/2508.04403) (2025) | Predicts how an utterance will end and prefetches the response | Within one turn | 1 |
| PASTE, TomasuLLM, Speculative Actions, Speculate with Memory, Interactive Speculative Planning | Speculative next action or tool | Action level | 0–1 |
| MemGPT, EM-LLM, A-MEM, Mem0 | Reactive memory | No anticipation | 0 |

Unverified: "Intention-aware Long-Context KV Cache Compression" (ACL 2026; the PDF would not parse).

**Slice verdict:** partially covered.

## Overall verdict: PARTIALLY COVERED, with a narrow open niche

Each component is published:
- topic-predicted idle-time prefetch of *text* (VoiceAgentRAG, ProAct)
- topic-segmented conversation KV selected *after* the query arrives (EpiCache)
- predicted-query → precomputed KV, cross-session (PerCache)
- lookahead KV prefetch keyed on agent workflow (PBKV, CacheScout)
- think-time warmup of *known* content (llm-d #2949)

Not found: **during inter-turn think time, predict the upcoming sub-queries from the active topic and promote the
matching topic-segmented KV of the conversation itself into a bounded GPU budget, ahead of the query.**

Constraints this sets for the work:
1. Required baselines: reactive episode selection (EpiCache-style), reload-everything during think time
   (llm-d / HyMCache-style), on-demand reload, and an oracle prefetch.
2. Must answer [2609.16215](https://arxiv.org/abs/2609.16215): prefetch failed to pay for its bandwidth.
   Report bytes moved, wasted prefetch, and TTFT together.
3. Selective prefetch only matters when **conversation KV > GPU budget**. If everything fits, or reloading
   everything fits within think time, "reload all" wins trivially. The experiment must sweep this regime.
