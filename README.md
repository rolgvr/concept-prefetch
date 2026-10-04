# concept-prefetch

Can an LLM serving system guess what you are about to ask, and have the right context loaded before you ask it?

This repo tests one version of that idea. During a conversation, the user spends a few seconds reading and typing
between turns, and the GPU sits mostly idle. If the system can predict which pieces of context the next question
will need, it can move them into GPU memory during that gap. The follow-up then starts faster.

The idea came from thinking about how recall works without mental imagery (aphantasia). The knowledge is always
there but nothing is rendered by default. You decide what you want before anything appears, and when a topic comes
up, related material gets "warmed" in the background so later questions about it feel instant. This project asks
whether that last part translates into a useful caching policy for LLMs.

## Short answer so far

- **Mechanism: works, but only under the right conditions.** Prefetching cuts follow-up latency when a cache miss is
  expensive (the context has to be recomputed, not just copied) and the next topic is predictable.
- **Real data: no gain on coding-agent sessions.** On my own coding-assistant session logs, nothing beat
  "keep what was used most recently" by more than ~2 points.
- **Still open: whether real human conversations drift predictably enough to matter.** That is the next experiment.

## Is this new?

Partly. I went through about 60 papers (full list in [notes/novelty.md](notes/novelty.md)). The closest work:

| Work | What it does | What's different here |
|---|---|---|
| [VoiceAgentRAG](https://arxiv.org/abs/2603.02206) (2026) | Predicts follow-up topics in a voice conversation and pre-fetches documents between turns | Fetches text chunks; this project prefetches the model's own KV cache |
| [EpiCache](https://arxiv.org/abs/2509.17396) (2025) | Splits a long conversation's KV cache into topic "episodes" | Picks the episode after the question arrives; here it's picked before |
| [PerCache](https://arxiv.org/abs/2601.11553) (2025) | Predicts future queries on a phone and precomputes their KV overnight | Works across sessions, not during a live conversation |
| [SYNAPSE](https://arxiv.org/abs/2601.02744) (2026) | Spreading-activation memory for agents | Retrieval only, triggered by the query |
| [Where should the KV cache live?](https://arxiv.org/abs/2609.16215) (2026) | Studies KV placement across GPU/CPU/SSD | Finds that history-based prefetch doesn't pay for its bandwidth, a result any prefetch scheme has to beat |

Predicting the topic, then moving that topic's KV segments onto a memory-limited GPU during the user's think time,
is the part I couldn't find done elsewhere.

## Experiment 1: toy prefetch on a small model

**Setup.**
- Model: Qwen2.5-0.5B-Instruct on an RTX 4090 laptop GPU.
- Data: 8 made-up companies, each with a fact sheet of 20 attributes. Each sheet is prefilled into its own KV
  segment (~770 tokens), kept in CPU memory.
- GPU budget: k segments at a time.
- Conversations: 40 simulated, 12 questions each. They mostly stay on one company and sometimes move to a "partner"
  or "competitor" named in its fact sheet.
- Policies compared at the same budget:
  - **on-demand:** load after the question arrives, keep nothing.
  - **reactive LRU:** load after the question, keep the most recent k.
  - **topic:** between turns, preload the current company plus its linked companies.
  - **oracle:** always preloads exactly the right one.

All policies run on the same turns in shuffled order. The laptop GPU switches between two clock speeds, and running
the policies one after another produced fake differences.

**Results.** Mean time to first token on follow-up questions, when a miss means recomputing the segment:

| GPU budget | on-demand | reactive LRU | topic | oracle | topic vs LRU |
|---|---|---|---|---|---|
| 1 segment | 72.0 ms | 48.8 ms | 49.6 ms | 33.9 ms | no difference |
| 2 segments | 72.0 ms | 46.7 ms | 43.8 ms | 33.1 ms | 6.2% faster (95% CI 2.6–10%) |
| 3 segments | 72.0 ms | 44.5 ms | 35.8 ms | 33.6 ms | 19.4% faster (95% CI 14–25%) |

![Latency and hit rate by GPU budget](results/ttft_hit.png)

**What this shows.**
- **The gain is just hit rate times miss cost.** When a miss only means copying 9 MB from CPU memory (~0.5 ms), no
  policy beats any other. Prefetching only matters when misses are expensive: large models, long contexts, or
  context stored on disk or across a network.
- **The advantage comes entirely from the topic links.** When the simulated conversations ignore the links, topic
  prediction and plain recency hit equally often (71% vs 70%). I wrote both the link structure and the predictor,
  so this toy shows the mechanism works but says nothing about real conversations.
- **Most prefetches are wasted.** 60–68% of preloaded segments were never used.
- **Side finding on accuracy.** Fact sheets encoded separately confuse the model when several are visible at once:
  97.5% accuracy with one sheet, 42.5% with two, 12.5% with all eight. Normal full-context prompting got 67.5%. So
  every policy attends only to the sheet the question names, which also means this experiment measures latency,
  not accuracy.

## Experiment 2: environment cues in real coding sessions

The second hypothesis broadens the first. People get sudden associations from what's around them, not only from the
conversation. For a coding assistant, "around" means things like a traceback, a grep result, or a file that was just
opened. The question: do those cues predict which files the next request will need, better than recency alone?

**Setup.**
- Data: my own coding-assistant session logs, parsed locally (raw logs never leave the machine). 5 sessions,
  1,018 turns.
- For each turn, the parser records:
  - **needed:** the files the assistant opened or edited;
  - **environment cues:** file paths that appeared in tool output;
  - **conversation cues:** file names mentioned in messages.
- Predictors:
  - **recency;**
  - **conversation cues only;**
  - **environment cues only;**
  - **graded activation:** a decaying score that combines all three.
- Measure: hit rate at a budget of k files.

**Results.** The pre-registered test (tune on the earliest sessions, score on the rest) gave 11.0% for graded
activation against 10.5% for recency at k=4. The bar was +10 points, so the test fails. With only five sessions the
tuning split was nearly empty. As a fairer check, I tuned on all the data, which flatters the method:

| GPU budget | recency | graded (tuned on all data) | gain |
|---|---|---|---|
| 2 files | 7.2% | 10.0% | +2.7 pt |
| 4 files | 10.5% | 12.3% | +1.9 pt |
| 8 files | 15.5% | 17.4% | +1.9 pt |

**Why it doesn't work here.**
- **The cues never run ahead of recency.** No file that showed up in tool output was later needed without already
  having been opened (0.0%).
- **Most needs are unpredictable.** 56% of the files each request needed had never appeared earlier in the session.
- **The agent consumes the cues itself.** In agentic coding, the assistant acts on the environment inside a single
  turn, so by the time the user types the next request, the cues have been used up. This is a clear negative result
  for between-turn prefetch in coding agents, from one person's logs.

## Limitations

- **Experiment 1 is built to favour the idea.** I designed both the conversation structure and the predictor.
- **The model is small.** A 0.5B model makes cache misses cheap, so the realistic "copy from CPU" case can't show a
  gain. The interesting regime is 7B+ models with long contexts.
- **Files stand in for KV segments in Experiment 2.** It is also one user and five sessions.
- **The wrong metric was pre-registered.** I locked in the median latency. Because the GPU clock is bimodal, the mean
  turned out to be the reliable statistic. Both are reported.

## Next

1. **Test predictability on data where a person drifts between topics.**
   - [TopiOCQA](https://huggingface.co/datasets/McGill-NLP/TopiOCQA): conversational QA whose topic switches follow
     Wikipedia links.
   - [ProCodeBench](https://arxiv.org/abs/2605.05700): real IDE interaction traces.

   This needs no GPU. If topic or environment prediction can't beat recency by about 10 points there, the idea stops.
2. **Only if step 1 passes, run the systems experiment properly.** That means:
   - an 8B model;
   - an expensive miss tier;
   - EpiCache- and VoiceAgentRAG-style baselines at the same memory budget;
   - accuracy measured per turn.

## Repo layout

```
DESIGN.md                  hypotheses, metrics and thresholds, committed before each experiment ran
toy/                       Experiment 1: data generator, segmented KV engine, runs, analysis
replay/                    Experiment 2: replay simulator, log parser, tests
results/                   aggregate results and the plot
notes/novelty.md           literature review with per-paper notes
notes/go-no-go.md          decision notes after each experiment
notes/log.md               lab notebook, including dead ends
```

## Reproduce

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install transformers accelerate numpy pandas matplotlib

# Experiment 1
.venv/Scripts/python toy/run.py              # topic-structure sweep and accuracy check
.venv/Scripts/python toy/run_interleaved.py  # latency measurement, all policies interleaved
.venv/Scripts/python toy/analyze.py

# Experiment 2 (reads local session logs; output stays in data/private/)
.venv/Scripts/python replay/parse_claude_logs.py
.venv/Scripts/python replay/replay.py data/private/claude_sessions.json --k 4
.venv/Scripts/python replay/replay.py data/private/claude_sessions.json --upper-bound
```
