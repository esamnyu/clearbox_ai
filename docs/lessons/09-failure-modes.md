# Lesson 9 — Failure Modes and What an Honest Writeup Looks Like

A Fellows interviewer will not probe whether you got a number. They will probe whether you know **why** the number can be wrong, and how you protected yourself from getting fooled. This lesson is the running checklist of every way the bench can mislead, and the discipline that makes a portfolio writeup defensible.

## Failure mode 1 — Small sample size

The codebase's `extract_steering_vector` runs on 8 contrastive pairs out of the box. Eight. For sentiment that is enough to demonstrate the recipe; for a portfolio claim about refusal it is not. JailbreakBench has 100 prompts; the refusal-direction literature uses 64–128 contrastive pairs after length-matching.

The failure mode: with $n=8$, the mean residual is dominated by which specific 8 prompts you happened to pick. The "sentiment direction" varies wildly run-to-run; an interviewer who rerolls your dataset gets a different vector. Demo demonstrations are *not* benchmark results.

**Protection:** report $n$. Bootstrap. If you cannot rerun the same experiment with $n$ resamples and report the variance, the result is a vibe, not a finding. The codebase's probe pipeline uses `random_state=42` for reproducibility *within* a run; the writeup must also report variance *across* runs.

## Failure mode 2 — Prompt-distribution mismatch

Extract the refusal direction on JailbreakBench (which leans toward weapons, hacking, and explicit instruction-following harms). Test on a held-out set drawn from XSTest (which includes a different harm distribution — slurs, self-harm prompts, social manipulation). The direction works less well, because the JailbreakBench-extracted vector is biased toward the JailbreakBench category distribution.

The failure mode: claims of the form "we identified the refusal direction" are dataset-relative. They should be claims of the form "the JailbreakBench-extracted refusal direction at layer 9 of Llama-3.2-1B reduces refusal rate on held-out JailbreakBench prompts by X."

**Protection:** name your dataset. Pin the commit hash of JailbreakBench. Report cross-dataset transfer numbers (extract on JBB, test on XSTest or HarmBench) explicitly.

## Failure mode 3 — Length effects

The contrastive pairs in `backend/refusal_pairs.py` are length-matched to within 20% on the Llama-3.2-1B tokenizer (see the file's docstring). If you skip this step, the difference-of-means vector contains a **length component** — longer prompts have different positional encodings and different last-token contexts than shorter ones, and those differences accumulate in the mean.

The failure mode: ablating your "refusal" direction also affects the model's response to long vs short prompts, in ways unrelated to refusal. The behavioral change you observe is partially a length artifact.

**Protection:** length-match. Report token-count distributions for both classes. If a reviewer asks "what is the median-length-difference between your harmful and harmless sets?", the answer should be a number you have on hand.

## Failure mode 4 — Layer selection by post-hoc maximization

You sweep layers 0–15 on Llama-3.2-1B, find that ablation at layer 12 maximizes refusal-rate drop, and report the layer-12 number. This is **selecting on the dependent variable** and inflates effect sizes.

**Protection:** either (a) use a principled selection criterion like COSMIC (Lesson 8) on a *held-out* sweep set, or (b) report all layers. Do not show only your best layer without naming the selection procedure. Anthropic Fellows reviewers explicitly call this out — they want a layer-sweep curve, not a single number.

## Failure mode 5 — Confusing "refusal dropped" with "safety dropped"

This is Lesson 7's whole point. If $\Delta$RefusalRate is 80 percentage points but $\Delta$AUC on the harmfulness probe is essentially zero, you removed the **verbal** refusal but the **harmfulness representation** is intact. Reporting "we disabled refusal" without the second axis is dishonest.

**Protection:** always run both. Always plot both. If your blog post says "refusal rate dropped from 78% to 22%," the very next sentence must say "harmfulness probe AUC dropped from 0.93 to 0.89." If the AUC didn't drop, **say so prominently** and frame the result accordingly: not "we removed safety," but "we removed the verbal refusal channel; safety representation persists." That framing is the one that gets a paper read, and the one that a reviewer wants you to volunteer.

## Failure mode 6 — Probe-as-evidence collapsing

A linear probe finds a hyperplane that separates harmful from harmless residuals. The probe's weights $w$ are a direction — but are they *the* direction the model uses for harmfulness judgment, or a direction that happens to discriminate this specific dataset's harmful from harmless examples (length, vocabulary, sentence structure)?

[Hewitt and Liang (EMNLP 2019)](https://aclanthology.org/D19-1275/) showed probes can fit random noise if given enough capacity. [Belinkov (2022)](https://arxiv.org/abs/2102.12452) is the survey. **Linear probes on residuals are evidence for "linearly available information," not evidence for "the model uses this information."**

**Protection:** intervene, don't just probe. The codebase combines them deliberately: the probe is the **measurement**, the projection-removal hook is the **manipulation**. Probe-alone is correlational; probe-plus-intervention is causal. Cite Hewitt-Liang in the writeup.

## Failure mode 7 — Generation variance

`ablate_along_direction` (line 497 of `research.py`) samples with `temperature=0.7, do_sample=True`. Run it twice on the same prompt and you get two different generations. A 1-prompt demo "before / after" is a single sample of a noisy process.

**Protection:** generate $k$ times per prompt with different random seeds and report the **distribution** of refusal labels. The UI's side-by-side is a hero demo, not a benchmark. The benchmark in `backend/refusal_bench/` should run hundreds of samples and bootstrap CIs.

## Reproducibility — the minimum

A portfolio reviewer needs to rerun your experiment and get your number. That requires:

1. **Model + commit pinning.** "Llama-3.2-1B-Instruct, HuggingFace revision `abc123...`." Not just "Llama-3.2-1B."
2. **Dataset pinning.** JailbreakBench at a specific commit. Alpaca subset with the seed used for sampling and the row indices selected.
3. **TransformerLens version.** Hook names change across major versions; the codebase's `requirements.txt` pins this.
4. **Random seeds.** For the probe (`random_state=42`), for generation sampling (`torch.manual_seed`), for the train/test split.
5. **A shareable URL.** The deploy goal (HF Spaces backend + Vercel frontend; see `docs/RESEARCH_LANDSCAPE_2026.md` §4) is exactly so that an interviewer can click one link, see the bench rerun, and confirm the numbers. A `live demo URL` in the application is worth more than a PDF with the same numbers.

## What a Fellows reviewer wants to see acknowledged

Not exhaustive, but the bar:

- **The novelty is the small-Llama replication, not the technique.** Zhao 2025 did the original dissociation on 8B+. The contribution is doing it on 1B — accessible to independent researchers — and showing whether the dissociation holds at that scale. Lead with this; do not overclaim.
- **Sample size and CIs reported throughout.** If you don't have them, get them before submitting.
- **Cross-technique comparison.** The Lesson 8 six techniques each map to a point on the Lesson 7 bench. If you have only one point (Arditi), you have a replication, not a contribution. If you have three or more, you have a methodology critique.
- **One specific surprise.** "We found that on Llama-3.2-1B, the COSMIC-selected layer disagrees with the refusal-token-heuristic-selected layer; the COSMIC layer gives a larger $\Delta$RefusalRate but no larger $\Delta$AUC." A single, specific, falsifiable mini-finding. That is the portfolio gold standard.
- **What you did not do, and why.** Honesty about scope. "We did not run the Maskey decomposition because we don't have an over-refusal contrastive dataset yet." Reviewers grade on calibration. Hand-wave at scope and they suspect you would hand-wave at results.

## The standard, in one sentence

A portfolio writeup is honest if a reviewer who re-runs your code on their own machine gets the same numbers within your stated CIs, and the writeup names every choice (model rev, dataset commit, layer, seeds, $n$, probe regularization, sampling temperature) that could shift those numbers if changed.

This is achievable in 9 days. It is the discipline that separates "I built a tool" from "I shipped a finding."

## Check yourself

1. Pick a number you might report — "refusal rate dropped from 78% to 22%." List five things a reviewer might ask before believing it.
2. Why does length-matching contrastive pairs matter mathematically, not just stylistically?
3. The writeup says "we disabled refusal." A reviewer rephrases it: "you disabled the verbal refusal output; the harmfulness signal in the residual is intact." What is the right response — concede, defend, or restate?

## Read next

You are done with the curriculum. The next read is [Arditi 2024 (arxiv 2406.11717)](https://arxiv.org/abs/2406.11717) and [Zhao 2025 (arxiv 2507.11878)](https://arxiv.org/abs/2507.11878) cover-to-cover, with `backend/research.py` and `backend/refusal_bench/harmfulness_probe.py` open beside them. Everything in this curriculum is a way to read those two papers without being lost.
