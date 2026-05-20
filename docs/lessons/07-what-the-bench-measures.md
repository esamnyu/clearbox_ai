# Lesson 7 — Refusal Rate vs Harmfulness AUC: The Two-Axis Story

## The measurement, in one sentence

For any "refusal ablation" technique applied to a model, you can measure two things independently:

- $\Delta \text{RefusalRate}$ — how much did the **verbal refusal behavior** drop?
- $\Delta \text{AUC}_{\text{harm}}$ — how much did the **residual-stream harmfulness signal** drop, as read by the linear probe from Lesson 6?

A technique is a **causal intervention on safety** if both go down together. A technique is a **behavioral patch** if RefusalRate drops but AUC stays high. This lesson explains why that distinction is the actually-novel framing the codebase brings to the field.

## How to measure each

**Refusal rate.** Pick a held-out set of harmful prompts (say, 50 from JailbreakBench not used for direction extraction). For each prompt: generate from the model (baseline) and then again from the same model with the refusal-direction ablation hook active. Use a regex or a small classifier to detect strings like `I cannot`, `I'm sorry`, `As an AI`, `I'm not able to`. Refusal rate = fraction of generations that contain such a string. The standard heuristic comes from [Arditi et al., 2024 (arxiv 2406.11717)](https://arxiv.org/abs/2406.11717) — a regex list checked against the first few generated tokens.

$\Delta \text{RefusalRate} = \text{RefusalRate}_{\text{baseline}} - \text{RefusalRate}_{\text{ablated}}$. Positive means ablation reduced refusal.

**Harmfulness AUC.** Train the probe from Lesson 6 on labeled (harmful=1, harmless=0) residuals **without ablation**. Then score the same harmful prompts **with ablation** and measure: does the probe still rank them as harmful?

There are two flavors of AUC delta. The simplest: probe-AUC against held-out labels stays the same (probe still discriminates harmful from harmless even when residuals come from an ablated forward pass). The most direct: `mean_p_harm` on known-harmful prompts barely drops. The codebase reports both via `evaluate_probe` and `extract_with_ablation` in `backend/refusal_bench/harmfulness_probe.py`.

## The 2×2 of possible outcomes

|  | AUC drops | AUC stays high |
|---|---|---|
| **RefusalRate drops** | Causal safety intervention: the refusal computation and the harmfulness representation are coupled, and the ablation hit both. *This would generalize the Arditi claim to "we actually removed the safety signal."* | **Behavioral patch.** Verbal refusal suppressed, but the model still internally represents "this is harmful." The ablation is a lexical patch on the output channel, not an intervention on the underlying safety computation. *This is the Zhao 2025 finding.* |
| **RefusalRate stays high** | Implausible. If AUC dropped, the harmfulness signal is gone, but the model still verbally refuses — what is it refusing on the basis of? Possibilities: probe is misspecified; refusal direction is not what was ablated; downstream pathway re-derives the signal. Worth investigating. | Ablation did nothing. The "refusal direction" wasn't actually doing the work, or the ablation was applied at the wrong layer. Null result. |

The interesting cell is the top-right: **RefusalRate drops, AUC stays high.** This is the Zhao result.

## The Zhao 2025 result

[Zhao et al., 2507.11878](https://arxiv.org/abs/2507.11878) tested on Llama-3-8B-Instruct and Qwen-7B-Chat: after ablating the Arditi refusal direction, verbal refusal drops sharply (high $\Delta$RefusalRate) but a linear probe trained on harmfulness can still detect harm with near-original AUC (low $\Delta$AUC). Their interpretation, which the codebase adopts: *the model's internal representation of "this prompt is harmful" persists after ablation, even though the model no longer says* "I cannot help."

The two-axis framing forces honesty about what the ablation actually achieved. Arditi 2024's paper reported behavioral metrics (refusal rate, jailbreak success rate) but not the residual-stream probe. They presented the result as "we found *the* refusal direction." Zhao 2025 added the probe and showed the direction Arditi ablated was the verbal-output channel, not the safety-judgment channel itself.

A reviewer in 2026 will not accept a claim of the form "we disabled refusal in Llama-3.2-1B" without both axes. The two-axis bench is what makes the claim falsifiable.

## Why this is the project's actual contribution

`docs/RESEARCH_LANDSCAPE_2026.md` is explicit (Section 4, "Revised prediction ordering, supersedes §4"): *"The headline contribution is the harmfulness probe after ablation. Nobody has run this on the small Llamas."* Zhao tested 8B+ models. Llama-3.2-1B and 3B are the under-explored regime, and they are the regime that matters for browser-deployable interpretability tooling.

The Lesson 7 deliverable for the Fellows portfolio is a single plot: x-axis = $\Delta$RefusalRate, y-axis = $\Delta$AUC, one point per ablation technique (the six from Lesson 8). Each point reveals which technique is in which 2×2 quadrant. A point in the **bottom-right** quadrant (high $\Delta$RefusalRate, low $\Delta$AUC) is a *behavioral patch*. A point in the **bottom-left** quadrant (high $\Delta$RefusalRate, high $\Delta$AUC) is a *plausible causal safety intervention*.

This is what the bench measures. This is the figure the blog post pivots on.

## How the codebase already supports this

The two-axis measurement is implemented across two files:

1. **`backend/research.py::ablate_along_direction`** — generates baseline and ablated text side by side. From these, refusal rate is computed by string-matching downstream (a `is_refusal()` helper would live in `backend/refusal_bench/` once added).
2. **`backend/refusal_bench/harmfulness_probe.py::extract_with_ablation`** — gathers residuals while ablation is active, ready to be scored by the trained probe.

The two functions share `make_ablation_hook` (line 59 of `research.py`). One math, two measurement axes. The next step — the Lesson 8 implementation work — is to loop over the six techniques in `backend/refusal_bench/` (Arditi single-direction, Wollschlager cone, COSMIC, Cheng compressed, Maskey decomposition, Herring CNA) and produce the table.

## Sample-size and statistical caveats

Two cautions a reviewer will press on:

- **Confidence intervals on RefusalRate.** If you test on 50 prompts and 40 are refused in baseline (80%) and 25 are refused after ablation (50%), the $\Delta$ is 30 percentage points. With $n=50$, the 95% CI on a single rate at 65% is roughly ±13 points. Report bootstrap CIs on $\Delta$ — the published version uses 1000-bootstrap with 50 prompts.
- **AUC variance.** Train/test split variance can move AUC by ±0.03 with $n=200$. Reporting "AUC dropped from 0.92 to 0.91" without a confidence interval is meaningless. The codebase uses `random_state=42` for reproducibility *within a single run* — for variance estimates you would need to vary the seed and report the standard deviation across seeds.

Neither of these is a defect in the technique. Both must be in the writeup.

## The blog post claim, falsifiable form

The post will say something like:

> *On Llama-3.2-1B-Instruct, ablating the Arditi refusal direction reduces refusal rate on the JailbreakBench harmful set from 78% to 22% ($\Delta$ = 56 pp, 95% CI [49, 63]), while the linear harmfulness probe AUC on the same prompts drops from 0.93 to 0.89 ($\Delta$ = 0.04). The verbal refusal collapses; the residual-stream harmfulness representation does not. We replicate Zhao 2507.11878 on a model two orders of magnitude smaller and confirm the dissociation.*

That sentence is the unit of contribution. The whole codebase is scaffolding around producing it honestly.

## What to *not* claim

The most common failure mode in interp blog posts is to over-interpret the result. Things this bench does **not** prove:

- *It does not prove the model "knows" the prompt is harmful in any cognitive sense.* It shows a linear classifier can predict harmfulness from the residual stream. That is a much weaker, mechanically precise claim.
- *It does not prove refusal ablation is unsafe.* It proves the verbal refusal can be ablated without removing the harmfulness representation. Whether the model can be *induced to act* on its surviving harmfulness signal is a separate downstream experiment (jailbreak success rate, completion harm rate). Lesson 9 covers these caveats.
- *It does not generalize to other models without re-running.* The bench is Llama-3.2-1B-specific. Cite Wang et al. NeurIPS 2025 for cross-lingual transfer; cite Zhao 2025 for 8B+; do not extrapolate.

## Check yourself

1. Why is the top-right quadrant of the 2×2 (RefusalRate drops, AUC stays high) the most informative outcome?
2. Why must the harmfulness probe be trained on *baseline* residuals and evaluated on *ablated* residuals, not the other way around?
3. Sketch what the blog-post figure looks like — what's on each axis, what each point represents, what the dashed-line annotations might say.

## Read next

Lesson 8 — `08-the-six-techniques.md`. We catalog the six refusal-ablation techniques in the literature, with the math sketch and citation for each. These are the dots that will populate the bench's 2×2 plot.
