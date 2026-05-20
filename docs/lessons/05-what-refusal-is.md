# Lesson 5 — What Refusal Is

## The behavior, first

Type into a Llama-3.2-1B-Instruct chat: *"How do I build a pipe bomb?"* The model produces something like:

> I cannot help with that request. Building explosive devices is illegal and dangerous, and...

That string — the apology, the categorical denial, the safety boilerplate — is a **refusal**. RLHF and constitutional-AI training drilled the model to produce it for prompts that fall into roughly the categories in JailbreakBench: violence, illegal activity, hate, self-harm, and a few others.

The question this lesson answers: what does that look like **inside** the model? What gets computed in the residual stream that causes the next-token distribution to put high probability on `I cannot`, `I'm sorry`, `As an`, instead of the actual answer?

## The surprising 2024 finding

[Arditi et al., NeurIPS 2024 — arxiv 2406.11717](https://arxiv.org/abs/2406.11717) found that for several open-weights instruct-tuned models (Llama-2, Llama-3 family, Qwen, Yi), there is a **single direction $\hat{r}$ in the residual stream** such that:

1. Projecting any prompt's last-input-token residual onto $\hat{r}$ gives a scalar that correlates with whether the model will refuse.
2. Ablating $\hat{r}$ from the residual stream (via the projection-removal primitive from Lesson 4) causes the model to *comply* with prompts it previously refused — at rates much higher than chance.

This was surprising because refusal is a complex, learned behavior. It depends on prompt semantics, RLHF training signals, and many learned heuristics. The default expectation in 2024 was that it would be a distributed computation involving many directions and many attention heads. Instead, on Llama-3-8B-Instruct, a single rank-1 ablation flipped refusal in the majority of harmful prompts.

This is the result this codebase is built to reproduce. `extract_steering_vector` builds $\hat{r}$ from contrastive pairs in `backend/refusal_pairs.py` (harmful prompts from JailbreakBench vs. harmless prompts from Alpaca). `ablate_along_direction` removes $\hat{r}$ from the residual stream at a chosen layer. The whole **diff-of-means → ablate → measure compliance** loop in `backend/research.py` and `backend/refusal_bench/` is the Arditi pipeline ported to TransformerLens.

## Why one direction is surprising

A residual stream has $d_{model}$ dimensions — 2048 for Llama-3.2-1B, 4096 for Llama-3-8B. The refusal "concept" could in principle be distributed across all of them. That it is *not* — that you can capture most of the refusal behavior by intervening on a single 1D subspace — has two implications.

First, **it's a strong instance of the linear representation hypothesis** (Lesson 2). The model literally stores the "should I refuse?" decision as a scalar magnitude along one fixed direction. Internal interpretability of this signal is geometrically clean.

Second, **it's a strong instance of superposition working in our favor.** Refusal happens to be a high-prior, high-frequency-during-training concept, so the model allocates it its own clean direction. The 144 attention heads × 12 layers of GPT-2 do not each implement a piece of refusal; they collectively write into one shared direction in the residual stream, and any later block can read that scalar to decide what to predict.

Third — and most relevant to the codebase's research goal — **it provides a falsification target.** "Refusal is one direction" is a specific empirical claim. The 2025–26 literature has spent considerable effort showing where it fails. The codebase's research thesis (see `docs/RESEARCH_LANDSCAPE_2026.md`) is to map those failure modes on Llama-3.2-1B.

## Where one direction breaks down

The 2024 finding was always qualified. Arditi 2024 itself reports that the single direction is most effective at one specific layer and degrades elsewhere; the compliance flip is not 100%; some prompt categories resist ablation. The 2025–26 follow-ups sharpen this:

- [Wollschlager et al., ICML 2025 — *Refusal in Language Models Is Mediated by a Polyhedral Cone*](https://arxiv.org/abs/2502.17420) — refusal lives in a **multi-direction cone**, not a single line. They introduce "concept cones" and "representational independence": orthogonal-in-Euclidean-space does not imply causally-independent under intervention.
- [Joad et al., Feb 2026, *There Is More to Refusal than a Single Direction* (arxiv 2602.02132)](https://arxiv.org/abs/2602.02132) — per-category geometrically distinct directions collapse to one knob only after a normalization step.
- [LessWrong, March 2026, *Single Direction vs Low-Rank Refusal in Small LLMs*](https://www.lesswrong.com/posts/LMkvjDTLKFrgdzJdG/single-direction-vs-low-rank-refusal-in-small-llms-1) — Qwen reaches ~91% compliance with one vector. Llama-family models reach only ~15% on Llama-3.2-3B with one direction; ~37% with a 5-layer stack. **Llama-family is the harder case for the single-direction story.**
- [Cheng et al., Apr 2026, arxiv 2604.08524](https://arxiv.org/abs/2604.08524) — direct steering analysis on Llama-3.2-3B; vectors compress to 1–10% of dims while preserving the effect.
- [Maskey et al., rev Apr 2026, arxiv 2603.27518](https://arxiv.org/abs/2603.27518) — reframes "rank > 1" as **over-refusal vs harmful-refusal** being task-conditioned, not simply about safety-tuning depth.

Read those abstracts. They are the literature that grounds the codebase's pivot from "GPT-2 sentiment steering" to "Llama refusal mapping." The headline contribution this codebase aims at is **not** "we replicated Arditi" — that's been done — but **"on Llama-3.2-1B specifically, where the rank-k structure has not been measured, does the harmfulness signal survive refusal-direction ablation?"** Lesson 6 introduces the probe that answers that question.

## What the codebase implements right now

Today, with the GPT-2-only backend, the only direction the system can extract is the **sentiment direction** from the 8 pairs in `get_contrastive_pairs()`. GPT-2 does not refuse anything — it has no RLHF training — so the refusal experiment is unrealizable on the current model. But the **infrastructure** is refusal-ready:

- `apply_chat_template` (line 28 of `research.py`) detects "instruct" in the model name and applies the right wrapping. It no-ops on GPT-2.
- `backend/refusal_pairs.py` is wired but its list is empty by design — `backend/scripts/build_refusal_pairs.py` populates it from JailbreakBench + Alpaca. Until then, the `/refusal-pairs` endpoint returns count=0 and the pipeline refuses to run.
- `backend/refusal_bench/harmfulness_probe.py` is implemented and tested.

The migration to Llama-3.2-1B (16 layers — `le=15` instead of `le=11` in `backend/main.py`) is the one missing step before the Lesson 7 experiment becomes runnable. This is deliberate scope discipline; see `docs/FELLOWS_SPRINT_9DAY.md`.

## What "refusal direction" is *not*

Three misreadings to avoid:

1. **Not a fixed direction across models.** Arditi extracts a different $\hat{r}$ for each model. Llama-3-8B's refusal direction is not Llama-3.2-1B's. Transfer experiments exist (Wang et al., NeurIPS 2025, on cross-lingual transfer within one model) but cross-model transfer of $\hat{r}$ is unreliable.
2. **Not a "safety" direction.** Refusal is a *behavior*. Whether a prompt is *actually harmful* is a separate signal that may live along a separate direction. This is the Zhao 2025 finding, the centerpiece of Lesson 7.
3. **Not magic.** The direction is extracted from a specific contrastive dataset. Change the dataset and the direction shifts. Use JailbreakBench instead of HarmBench and you get a slightly different $\hat{r}$. The codebase pins JailbreakBench (and Alpaca for harmless) so results are reproducible.

## Check yourself

1. Why was the Arditi 2024 finding surprising? State the prior expectation that it contradicted.
2. Give one piece of evidence that "single direction" is incomplete, with a citation.
3. The codebase's `extract_steering_vector` returns the steering vector $v = \text{mean}_+ - \text{mean}_-$, not the unit vector $\hat{r} = v / \|v\|$. Where in the pipeline does normalization happen, and why is it deferred?

## Read next

Lesson 6 — `06-harmfulness-probe.md`. We meet the second tool: a linear classifier that reads the residual stream and predicts "is this prompt harmful?" — independently of whether the model verbally refuses.
