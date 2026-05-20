# Lesson 8 — The Six Techniques

Each section below is the math sketch and one-paragraph "what is it doing" for a refusal-ablation technique that the bench from Lesson 7 evaluates. The current codebase implements technique 1 (Arditi) fully and provides the infrastructure to extend to the rest. The arxiv IDs are the canonical citations.

## 1. Arditi — Single Direction

[Arditi et al., NeurIPS 2024 — arxiv 2406.11717](https://arxiv.org/abs/2406.11717)

**The technique.** Extract a refusal direction $\hat{r}$ via difference-of-means over JailbreakBench (harmful) vs Alpaca (harmless) prompts, at the last-input-token position of an instruct chat template. Ablate at one chosen layer via the projection-removal primitive:

$$h' = h - (h \cdot \hat{r}) \hat{r}$$

**What it claims.** A single 1D subspace of the residual stream causally mediates refusal in Llama-2-7B-Chat, Llama-3-8B-Instruct, Qwen-1.5-Chat, and Yi-6B-Chat. Removing it flips refusal to compliance on the majority of harmful prompts.

**Codebase status.** Fully implemented. `extract_steering_vector` (with refusal pairs) → `ablate_along_direction`. The only missing piece on the GPT-2 backend is JailbreakBench pairs (run `backend/scripts/build_refusal_pairs.py`). On Llama-3.2-1B, also bump `le=11` to `le=15` in `backend/main.py`.

**What the bench expects to find on Llama-3.2-1B.** Per the LessWrong March 2026 replication on Llama-3.2-3B (~15% compliance with one vector), single-direction ablation on 1B should be **partial** — significant $\Delta$RefusalRate but not the near-100% flip seen on Qwen. Per Zhao 2025, the harmfulness AUC should stay roughly flat — single-direction ablation removes the verbal-refusal channel without removing the harmfulness representation.

## 2. Wollschlager — Polyhedral Cone

[Wollschlager et al., ICML 2025 — arxiv 2502.17420](https://arxiv.org/abs/2502.17420)

**The technique.** Refusal is mediated by a **polyhedral cone** of $k$ directions $\{\hat{r}_1, \ldots, \hat{r}_k\}$ that are mechanistically independent under intervention. Extraction iterates: extract direction 1 via difference-of-means; ablate it; re-extract on the residual to get direction 2 orthogonal-after-ablation; repeat. Ablation removes the projection onto the entire cone:

$$h' = h - \sum_{i=1}^{k} (h \cdot \hat{r}_i) \hat{r}_i$$

assuming the $\hat{r}_i$ are made orthonormal (Gram-Schmidt during extraction).

**The key conceptual contribution.** "Representational independence": two directions can be orthogonal in Euclidean space yet causally coupled under intervention (e.g., a downstream component reads $\hat{r}_1 + \hat{r}_2$). The cone is the set of directions that are *causally* independent — ablating any subset preserves the predicted compliance effect of ablating just that subset.

**What it claims.** On Llama-3-8B-Instruct, a polyhedral cone of $k \approx 3$–$5$ orthogonal directions achieves substantially higher compliance than the single-direction baseline. On Llama-3.2-3B (per LessWrong replication), 5 layers/stacks bring compliance from ~15% to ~37%.

**Codebase status.** Not yet implemented. Adding it = extend `extract_steering_vector` to return a list of vectors via iterative residualization, and extend `make_ablation_hook` to subtract a sum-of-projections (the `coeffs` line becomes a matrix multiply against an orthonormal basis of $k$ directions).

## 3. COSMIC — Cosine-Objective Auto-Selection

[Luu et al., ACL 2025 Findings — arxiv 2506.00085](https://arxiv.org/abs/2506.00085)

**The technique.** A heuristic-free way to pick the **layer** and the **direction** for refusal ablation. Instead of sweeping over layers and using the refusal-token logit difference as the selection criterion (Arditi's method), COSMIC scores candidate (layer, direction) pairs by **cosine similarity between a refusal-template embedding and the candidate ablated direction**. Pick the pair that maximizes the cosine criterion. The ablation operation is identical to Arditi's $h - (h \cdot \hat{r}) \hat{r}$; the contribution is in how $\hat{r}$ and the layer are chosen.

**Why it matters for the bench.** The "which layer is best?" question dominates ablation result variance. COSMIC removes the manual sweep. If you report Arditi numbers without naming your selection procedure, a reviewer will ask "did you sweep?" and the answer needs to be "yes, by criterion X." COSMIC is the no-heuristic answer.

**Codebase status.** Not implemented. Adding it = a `select_refusal_direction_cosmic` function that takes the contrastive pairs and returns the optimal `(layer, direction)` tuple under the cosine objective. About one day of compute to wire in.

## 4. Cheng — Compressed Steering Vectors

[Cheng et al., April 2026 — arxiv 2604.08524](https://arxiv.org/abs/2604.08524)

**The technique.** Direct steering analysis on Llama-3.2-3B shows the refusal "direction" extracted from difference-of-means is actually a dense vector in $\mathbb{R}^{d_{model}}$ that **can be compressed to 1–10% of its non-zero dimensions** while preserving the behavioral effect. Concretely: apply $L_1$ regularization (or top-$k$ truncation by absolute value) to the steering vector, normalize, and ablate the sparse direction. Behaviorally indistinguishable from the dense version; mechanistically isolates an OV-circuit pathway.

**Why it matters.** Sparsity = interpretability. A dense 3072-dim direction is uninterpretable. A direction with 50 non-zero coordinates is potentially tractable — you can ask which residual-stream coordinates carry the refusal signal and tie those to specific attention head outputs.

**Codebase status.** Not implemented. Adding it = a `compress_direction(v, k)` helper that zeros all but the top-$k$ coordinates of $v$ and renormalizes. Trivial to write. The interesting work is the analysis: report the compliance-vs-sparsity curve and check whether the OV-circuit attribution holds on 1B.

## 5. Maskey — Over-Refusal vs Harmful-Refusal Decomposition

[Maskey et al., rev April 2026 — arxiv 2603.27518](https://arxiv.org/abs/2603.27518)

**The technique.** The "rank > 1 refusal subspace" finding from Wollschlager is reframed as a decomposition into two qualitatively distinct subspaces: an **over-refusal** subspace (the model refuses prompts that look like harmful prompts but are actually benign — e.g., "How do I delete a file in Linux?" being misread as adversarial) and a **harmful-refusal** subspace (the model refuses genuinely harmful prompts). The two subspaces are extracted from two different contrastive datasets: harmful-vs-harmless for the second, and benign-but-refused-vs-benign-and-complied for the first. Ablate either independently.

**Why it matters.** It explains why "rank > 1" doesn't necessarily mean "safety tuning is deeper than one direction." It might mean *two different things were folded into one refusal label* during RLHF. Lesson 7's bench should plot Maskey's two ablations as separate points: ablating over-refusal should restore compliance on falsely-refused benign prompts without affecting harmful-prompt refusal.

**Codebase status.** Requires an over-refusal contrastive dataset (e.g., from [OR-Bench](https://arxiv.org/abs/2405.20947)). Not implemented; needed only if the project extends past the headline experiment.

## 6. Herring — Contrastive Neuron Ablation (CNA)

Cited in `docs/RESEARCH_LANDSCAPE_2026.md` as 2605.12290 (April 2026 preprint).

**The technique.** Instead of ablating a *direction* in the residual stream, ablate the contribution of *specific MLP neurons* that fire differently on harmful vs harmless prompts. The selection criterion is the per-neuron difference-of-means activation; ablation zeros those neurons' outputs in the forward pass. The intervention is at the **MLP-out** layer, not the post-block residual stream.

**Why it matters.** This is a *non-projection-removal* causal intervention. It tests a different mechanistic hypothesis: maybe refusal is mediated by specific MLP features, not by a smooth residual-stream direction. If CNA matches Arditi's $\Delta$RefusalRate but at a sparser set of locations, the "single direction" interpretation is incomplete — the direction was a low-rank summary of a sparse neuron pattern.

**Codebase status.** Requires a new hook on `blocks.{layer}.mlp.hook_post` rather than `hook_resid_post`. The selection of which neurons to ablate is a separate step (top-$k$ by per-neuron mean activation difference). About a day of new infrastructure to add. Worth doing only after the Arditi headline result is shipped.

## How the bench combines them

Each technique produces a $(\Delta\text{RefusalRate}, \Delta\text{AUC}_{\text{harm}})$ point on the Lesson 7 plot. The expected layout, based on the literature:

- **Arditi (single direction):** moderate $\Delta$RefusalRate, near-zero $\Delta$AUC. (Behavioral patch.)
- **Wollschlager (cone):** higher $\Delta$RefusalRate, slightly larger $\Delta$AUC.
- **COSMIC:** same shape as Arditi, with better layer selection — point shifts further right.
- **Cheng (compressed):** same as Arditi behaviorally, but more interpretable.
- **Maskey (decomposed):** two points — one for the over-refusal ablation (small $\Delta$RefusalRate on harmful prompts, large on benign prompts), one for harmful-refusal ablation.
- **Herring (CNA):** unknown — this is the empirical question that makes the bench worth running.

The plot is the writeup. The plot fits in 800 words. The 9-day sprint produces the plot.

## Check yourself

1. For each of the six techniques, state in one sentence what mathematical object it ablates and at which hook point it intervenes.
2. Why is COSMIC a methodology paper, not a finding paper? Where does it fit in a portfolio writeup?
3. Maskey reframes "rank > 1" as a property of *what was labeled refusal* rather than *how deep safety tuning is*. What's the policy implication if Maskey is right?

## Read next

Lesson 9 — `09-failure-modes.md`. The honest writeup. What each technique cannot tell you, where it can be gamed, and what the portfolio reviewer wants to see acknowledged.
