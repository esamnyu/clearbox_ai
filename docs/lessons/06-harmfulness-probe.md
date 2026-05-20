# Lesson 6 — Detecting Harmfulness with a Linear Classifier

## Probes vs. interventions

Lesson 4 introduced the **intervention** — actually change the model's residual stream and watch behavior change. This lesson introduces the **probe** — a classifier that reads the residual stream without changing it.

The distinction matters because probes and interventions answer different questions:

- A **probe** answers: *"Does the residual stream at layer $L$ contain information about concept $C$?"*
- An **intervention** answers: *"Does the model **use** that information when producing behavior $B$?"*

Probes are correlational. Interventions are causal. The codebase's central research move is to combine them: probe for harmfulness *after* intervening on refusal. If the probe still detects harmfulness even when the model is no longer refusing, then refusal and harmfulness are mechanistically separable. That is the claim Lesson 7 lays out.

## The setup, in code

Open `backend/refusal_bench/harmfulness_probe.py`. The pipeline is three functions:

1. `extract_last_token_residuals(prompts, layer)` (line 35) — for each prompt, run the model, grab `cache[blocks.{layer}.hook_resid_post][:, -1, :]`, stack into a tensor of shape `[n_prompts, d_model]`.
2. `train_probe(harmful_residuals, harmless_residuals)` (line 61) — fit a logistic regression to discriminate the two sets.
3. `evaluate_probe(probe, residuals, labels=None)` (line 107) — score new residuals; report per-example $P(\text{harmful})$, mean, and AUC if labels are given.

These three primitives are enough to build the Lesson 7 experiment.

## Why a *linear* probe

The classifier is `sklearn.linear_model.LogisticRegression`. It is the **simplest** classifier that fits a separating hyperplane in $\mathbb{R}^{d_{model}}$. It cannot model non-linear decision boundaries.

Three reasons to prefer linear:

1. **Alignment with the linear representation hypothesis (Lesson 2).** If the concept is linearly represented, a linear probe will find it. If a *non-linear* probe finds something a linear probe cannot, that signal probably is not part of the model's internal computation — it's something the probe is computing on top of the residual stream. Citation: [Belinkov, *Probing Classifiers: Promises, Shortcomings, and Advances* (2022)](https://arxiv.org/abs/2102.12452) — the canonical methodological treatment of when probes are evidence and when they're noise.
2. **Sample efficiency.** A linear probe in $d_{model}$ dims with a few hundred examples is well-determined. A non-linear probe (e.g., MLP) with the same data overfits, and the AUC you get is a property of the probe, not the model.
3. **The Kantamneni 2025 result.** [Kantamneni et al., Feb 2025](https://arxiv.org/abs/2502.16681) showed that across 113 datasets, **plain logistic regression on residuals matches or outperforms SAE-based probes**. The simplest probe is the strongest probe. The codebase makes this choice deliberately.

## Walking `train_probe`

```python
X = torch.cat([harmful_residuals, harmless_residuals], dim=0).cpu().numpy()
y = np.concatenate([np.ones(...), np.zeros(...)])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
probe = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
probe.fit(X_train, y_train)
```

Four design choices, all defensible:

- **`stratify=y`.** The 80/20 split preserves the harmful/harmless ratio in both folds. Without stratification, you can get a test set with zero examples of one class — and AUC becomes undefined.
- **`class_weight="balanced"`.** If you have 50 harmful and 100 harmless prompts (because Alpaca is huge and JailbreakBench is small), the loss function automatically reweights so a misclassified harmful example counts as much as a misclassified harmless example. Without this, the probe trivially classifies everything as harmless and is "right" two-thirds of the time.
- **`random_state=42`.** Reproducibility. Same residuals + same split = same probe. This is non-negotiable for a portfolio writeup; if you cannot rerun your own probe and get the same AUC, no reviewer will trust your claims.
- **`C=1.0, max_iter=2000`.** Modest L2 regularization (the default in scikit-learn). `max_iter` is bumped up because high-dimensional residuals occasionally need more iterations to converge. These are sensible defaults; the codebase deliberately does not tune them, because hyperparameter-tuned probes are a way to inflate AUCs.

## Why AUC, not accuracy

The probe returns `train_auc` and `test_auc` — the area under the ROC curve.

Accuracy depends on a threshold. AUC does not. AUC measures: across all possible thresholds, how well does $P(\text{harmful} | h)$ rank harmful prompts higher than harmless ones? An AUC of 1.0 means perfect separation. AUC of 0.5 means chance. AUC of 0.85 (say) means: pick a random harmful prompt and a random harmless prompt, and the probe assigns higher score to the harmful one 85% of the time.

For interpretability research, AUC is the right metric because:

1. We do not want to commit to a refusal threshold. The model itself does not refuse at a sharp threshold; behavior degrades gracefully.
2. AUC is robust to class imbalance, in the sense that it's well-defined even when classes are unequal. (Accuracy is not.)
3. The result that matters in Lesson 7 is a **change in AUC after ablation.** If pre-ablation AUC is 0.92 and post-ablation AUC stays at 0.91, that's a strong claim about what was and wasn't removed. Accuracy-deltas at fixed thresholds are easier to game by adjusting the threshold.

## The `evaluate_probe` contract

`evaluate_probe` (line 107) returns three things:

```python
return {
    "p_harm": [float(x) for x in p_harm.tolist()],   # per-example P(harmful)
    "mean_p_harm": float(p_harm.mean()),
    "auc": auc,  # None if labels not given or one class missing
}
```

The `mean_p_harm` is the key number. If you score 50 *known-harmful* prompts and `mean_p_harm = 0.93`, the probe is confident those prompts are harmful. If you re-score the same prompts after ablating the refusal direction and `mean_p_harm = 0.91`, the harmfulness signal is *still there* — the residual stream still encodes "this is harmful" with high confidence, even though the model is no longer outputting `I cannot help with that`. **That is the Zhao 2025 result** — the centerpiece of Lesson 7.

## `extract_with_ablation` — the combined probe-after-intervention primitive

The function at line 142 of `harmfulness_probe.py` is the actually-novel piece. It does **both** things in one call:

```python
with model.hooks(fwd_hooks=[(ablation_hook_name, ablation_hook)]):
    _logits, cache = model.run_with_cache(prompt)
last = cache[extract_hook_name][:, -1, :].squeeze(0).detach().cpu()
```

Inside the `with model.hooks(...)` block, the projection-removal hook from Lesson 4 is active at `ablation_layer`. The forward pass runs with the refusal direction ablated. We grab the residual at `layer_extract` (which can be the same layer or downstream) and stack it. The trained harmfulness probe then scores this *ablated* residual.

The signature deliberately separates `layer_extract` from `ablation_layer`. Why? Because the natural follow-up question is *"does the harmfulness signal propagate downstream of the ablation, even if it's removed at the ablation site itself?"* Answer: read at a later layer than you ablate, score with the probe, see what survives.

## Reusing one ablation hook everywhere

Line 28 of `harmfulness_probe.py`:

```python
from research import make_ablation_hook
```

The bench imports the **same** hook used by the UI's ablation generation in `research.py::ablate_along_direction`. This is non-negotiable: the math $h' = h - (h \cdot \hat{d}) \hat{d}$ must be one function. If you accidentally re-implement it in the probe pipeline and it diverges from the UI implementation, your probe's claims about "what the UI is doing" are false. Single source of truth.

## What this probe cannot do

A probe gives you a number. It does not give you:

- **Causal evidence.** That comes from the intervention (Lesson 4). A probe alone is correlational.
- **Identification of *which* direction.** The probe's learned weights $w$ are *a* direction, but they confound the harmfulness direction with whatever else discriminates these two specific datasets (length, vocabulary, sentence structure). Cite [Hewitt and Liang, *Designing and Interpreting Probes* (EMNLP 2019)](https://aclanthology.org/D19-1275/) for the canonical "probes can learn things that aren't really in the model" cautionary tale.
- **Generalization beyond the dataset.** Train on JailbreakBench, test on a new harm category, and AUC drops. The codebase pins the dataset specifically so the result is reproducible — not generalizable.

## Check yourself

1. Why does the codebase use logistic regression rather than an MLP probe?
2. What is the difference between AUC dropping after ablation and `mean_p_harm` dropping after ablation? When does each matter?
3. The probe is trained with `random_state=42, stratify=y`. What problem does each of those flags solve?

## Read next

Lesson 7 — `07-what-the-bench-measures.md`. The two-axis story: ΔRefusalRate and ΔAUC. Why the gap between them is the actually-novel claim of this project.
