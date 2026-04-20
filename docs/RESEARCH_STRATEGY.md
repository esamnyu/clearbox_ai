# NeuroScope Research Strategy

> Rigorous replication and extension of the low-rank refusal finding in small Llamas
> Revised April 19, 2026

---

## Executive Summary

This strategy reframes NeuroScope as a **rigorous replication and extension** of a recent public claim (IvanC, LessWrong, March 2026) that refusal in Llama-3.2-3B-Instruct is **low-rank, not single-direction**. We confirm or refine the claim with held-out ablation, upgrade the measurement with harmfulness/refusal positional disentanglement (Zhao et al., 2025), and deliver a methodologically defensible writeup suitable as a portfolio or interview piece.

The January 2026 plan targeted refusal research using Arditi et al.'s single-direction methodology. Three months of field movement force a reframe: the single-direction hypothesis **does not hold on our target model**, and last-token caching conflates two distinct representations. This revision absorbs both findings and converts the biggest methodological threat into the actual research question.

### Key Recommendations

1. **Model**: Llama-3.2-3B-Instruct (unchanged — it is *because* refusal is low-rank here that the experiment is interesting)
2. **Target behavior**: Refusal, treated as low-rank (k=3) not single-direction
3. **Methodology**: Difference-of-means at k=1 (baseline) and SVD top-k (k=3), with directional ablation + addition as causal tests
4. **Positional disentanglement**: Cache activations at both instruction-end and post-instruction positions (Zhao 2025)
5. **Datasets**: JailbreakBench + Alpaca subset + SORRY-Bench held-out; StrongREJECT as automated evaluator
6. **Scope**: One complete experiment (Level B) with a stretch SAE-decomposition pass

---

## 1. Framing: Why Replication-Plus-Extension

### The Scientific Claim We Are Testing

IvanC (LessWrong, March 2, 2026) reports empirically that Llama-3.2-3B-Instruct exhibits a low-rank refusal structure: a single extracted vector yields only 15–26% compliance, while stacking the top-3 SVD directions from layers {9, 10, 7} pushes compliance to ~37% at 94% coherence. Llama-3.1-8B and Qwen-3-1.7B, by contrast, remain well-modeled by a single direction.

This is a blog post, not peer-reviewed work — which is exactly why it merits a rigorous, independent test. Our contribution:

1. **Confirm or refute** the k=3 > k=1 claim on a proper held-out split, with coherence and collateral damage measured.
2. **Disentangle** harmfulness from refusal (Zhao et al., arXiv:2507.11878) — an upgrade IvanC did not include.
3. **Characterize** where in the network low-rank structure emerges (which layers, which token positions).

### Why Not Switch to Llama-3.1-8B

Llama-3.1-8B would give a cleaner single-direction story and better SAE coverage via LlamaScope. We decline because:

- The low-rank finding *is* the story on 3B. Chasing a cleaner graph by switching models is survivorship bias.
- 8B needs ~16 GB VRAM; 3B fits 8 GB, which keeps local pair-programming tractable for Ethan.
- A rigorous 3B-specific replication is more defensible than a derivative 8B study.

---

## 2. Methodology

### Phase 1: Data Collection & Activation Caching

1. Run the model on 100 harmful instructions (JailbreakBench) and 100 matched harmless instructions (Alpaca, token-length-matched).
2. Cache residual stream activations at **two positions** per sequence:
   - **Instruction-end position** — hypothesized to encode harmfulness
   - **Post-instruction position** — hypothesized to encode refusal
3. Cache across middle-to-late layers (target range: 7–15 for Llama-3.2-3B).
4. Hold out 50 examples per class for validation; never touch during extraction.

### Phase 2: Direction Extraction at Two Ranks

For each (layer L, position P), extract both:

```
k = 1 (baseline):  r = mean(harmful_activations) − mean(harmless_activations)
k = 3 (primary):   r_1, r_2, r_3 = top-3 left singular vectors of the per-example
                   difference matrix [harmful_i − harmless_i]_i
```

The k=1 vs k=3 comparison is the central experiment.

### Phase 3: Causal Verification

For each rank condition, run two intervention tests on held-out prompts:

- **Necessity (ablation)**: Project residual stream orthogonal to the direction(s). Measure attack success rate on held-out harmful prompts.
- **Sufficiency (addition)**: Add the direction(s), scaled by α, to the residual stream on held-out harmless prompts. Measure induced refusal rate.

Score refusal with StrongREJECT, not keyword matching.

### Phase 4: Collateral Damage Measurement

Run the k=3 ablated model on:

- MMLU subset — report accuracy delta vs. unablated
- KL divergence vs. unablated model on 500 held-out Alpaca prompts
- XSTest — over-refusal on safe-but-sensitive prompts

Methodological rigor comes from reporting the cost alongside the gain.

### Phase 5 (Stretch): SAE Decomposition

If weeks 3–4 run under budget:

- Load publicly available SAEs for Llama-3.2-3B where available; otherwise skip.
- Project the top-3 refusal directions into SAE feature space.
- Report which features fire, with candidate semantic labels.

Community signal (secondhand, unverified) suggests DeepMind has deprioritized fundamental SAE research because linear probes outperform them on harmful-intent detection. We therefore treat SAEs as interpretability polish, not as the source of causal claims.

---

## 3. Datasets

| Dataset | Role | Size |
|---------|------|------|
| **JailbreakBench** | Primary harmful prompts | 100 behaviors |
| **Alpaca (subset)** | Matched harmless prompts | 100–200, length-matched |
| **SORRY-Bench** | Held-out evaluation across finer taxonomy | 440 across 44 topics |
| **XSTest** | Over-refusal evaluation | 450 prompts |
| **StrongREJECT** | Automated refusal evaluator (not a dataset) | — |

Changes from January 2026: **SORRY-Bench** added for finer-grained taxonomy vs JailbreakBench's flat 100 behaviors; **StrongREJECT** established as the canonical automated evaluator (outperforms keyword-based non-refusal detection).

### Dataset Quality Checklist

- Pairs differ only in harmfulness
- Token-length-matched per pair
- Chat-template-compliant (Llama-3.2 instruction format)
- Held-out splits for validation and collateral damage
- Never train and evaluate on the same JailbreakBench examples

---

## 4. Visualization Strategy

### Priority 1: k=1 vs k=3 Attack-Success-Rate Comparison

The headline plot. Directly shows the central claim. Bar chart or paired points per ablation strength.

### Priority 2: Per-Layer, Per-Position Direction Strength

At each layer, cosine similarity of the k=1 direction across positions (instruction-end vs post-instruction), plus magnitude per layer. Makes the Zhao et al. disentanglement visible.

### Priority 3: Collateral Damage Plot

MMLU delta and Alpaca-KL as a function of ablation strength. Honest framing.

### Priority 4 (if time): Head Activation Grid

Which heads fire most during refusal. Useful for interpreting what the low-rank subspace is doing.

### Defer

3D PCA trajectories and embedding spaces. Visually compelling but do not support causal claims.

---

## 5. Common Pitfalls to Avoid

### Methodological

1. **Zero ablation** — use mean ablation instead. Zero has no privileged meaning in activation space.
2. **Single-direction overconfidence** — *known wrong* for Llama-3.2-3B. Always report k=1 and k=3.
3. **Position conflation** — last-token caching conflates harmfulness and refusal (Zhao 2025). Cache at both positions.
4. **Collateral damage silence** — MLP ablation hurts capability more than attention ablation. Report KL, MMLU delta, and XSTest over-refusal explicitly.
5. **Insufficient held-out validation** — training on JailbreakBench and evaluating on JailbreakBench is a classic reviewer flag.

### Conceptual

1. **Correlation vs. causation** — intervention is required (ablation + addition), not just probe accuracy.
2. **Semantic overinterpretation** — the refusal subspace's "meaning" is unclear. Report observations, not assumed interpretations.
3. **Generalization leaps** — the claim is scoped to Llama-3.2-3B. Do not imply findings transfer without testing.

---

## 6. Answers to Open Questions

| Question | Recommendation | Rationale |
|----------|---------------|-----------|
| Q1: Model | **Llama-3.2-3B-Instruct** | Low-rank structure here *is* the research question |
| Q2: Target behavior | **Refusal (low-rank, k=3)** | Direct test of IvanC March 2026 finding |
| Q3: Dataset scope | **JB + Alpaca + SORRY-Bench held-out** | Held-out generalization across harm taxonomy |
| Q4: Real-time level | **On-submit (not streaming)** | Unchanged — lower complexity, sufficient |
| Q5: First visualizations | **k=1 vs k=3 ASR + collateral damage** | Support the central claim honestly |
| Q6: Depth level | **Level B with SAE stretch** | One complete experiment + optional polish |
| Q7: GPT-2 work | **Port patterns, do not maintain two tracks** | Unchanged |
| Q8: Runtime | **Python + TransformerLens v2.x** | v3.0 released Apr 17 2026; migrate after experiment |

---

## 7. Tooling

### Primary: TransformerLens v2.x, pinned

v3.0 released April 17, 2026 — two days before this revision. Debugging a fresh API migration (new `TransformerBridge` interface, legacy API deprecated) at the same time as new methodology is too many unknowns. Pin v2.x now; evaluate v3 migration as a follow-up project.

### Alternative to Evaluate Later

**nnterp** (arXiv:2511.14465, NeurIPS 2025 MI workshop) wraps `nnsight` to give a standardized interface across 50+ HuggingFace architectures without reimplementation — which sidesteps the numerical-mismatch risk TransformerLens carries. Worth a week-long spike in a v2 project.

### Hardware

Llama-3.2-3B fp16 fits comfortably in 8 GB VRAM. CPU/MPS fallback works for smoke tests but is too slow for full 100-prompt runs. A single mid-range consumer GPU is enough.

---

## 8. Revised Roadmap

### Week 1: Foundation

- **Ethan** — Pin TransformerLens v2.x; verify Llama-3.2-3B-Instruct loads via `HookedTransformer.from_pretrained`; implement dual-position activation caching; 10-example smoke test.
- **Moon** — Download JailbreakBench; sample Alpaca; token-length-match 100 pairs; verify model refuses all 100 harmful baseline; read IvanC post + Zhao et al. + Arditi et al.

### Week 2: Direction Extraction

- **Ethan** — Implement diff-of-means (k=1) and SVD top-k (k=3) extraction; expose both as FastAPI endpoints.
- **Moon** — Run extraction across layers 7–15 at both positions; identify best (layer, position) per rank; produce per-layer direction-strength plot.

### Week 3: Causal Verification

- **Ethan** — Implement ablation + addition hooks for both ranks; implement collateral damage harness (MMLU subset, Alpaca-KL, XSTest, StrongREJECT).
- **Moon** — Run all four intervention combinations on held-out sets; log ASR, coherence, MMLU delta, KL, over-refusal rate.

### Week 4: Analysis & Writeup

- **Ethan** — Build the three headline visualizations; if time, head activation grid.
- **Moon** — Interpret results; draft methodology and findings. Stretch: SAE decomposition of top-3 directions.

---

## 9. Key References

### Foundational (retained from January 2026)

- Arditi et al., NeurIPS 2024 — "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717) — the original single-direction methodology, which this project tests
- Heimersheim, 2024 — "How to Use and Interpret Activation Patching" — intervention best practices
- FAR.AI, Jan 2025 — "Open Problems in Mechanistic Interpretability"

### New (added April 2026)

- **IvanC**, LessWrong, March 2, 2026 — ["Single Direction vs Low-Rank Refusal in Small LLMs"](https://www.lesswrong.com/posts/LMkvjDTLKFrgdzJdG/single-direction-vs-low-rank-refusal-in-small-llms-1) — the specific claim being replicated
- **Zhao et al.**, 2025 — "LLMs Encode Harmfulness and Refusal Separately" ([arXiv:2507.11878](https://arxiv.org/abs/2507.11878)) — positional disentanglement
- **Wen et al.**, ICLR 2026 — "The Geometry of Refusal in LLMs: Concept Cones and Representational Independence" ([arXiv:2502.17420](https://arxiv.org/abs/2502.17420)) — multi-dimensional refusal
- **SOM Directions are Better than One**, AAAI 2026 ([arXiv:2511.08379](https://arxiv.org/abs/2511.08379)) — generalization of diff-of-means
- **O'Brien et al.**, EMNLP Findings 2025 — "Understanding Refusal in LMs with SAEs" — SAE-based decomposition
- **nnterp**, NeurIPS 2025 MI workshop ([arXiv:2511.14465](https://arxiv.org/abs/2511.14465)) — tooling alternative to TransformerLens

### Tools & Datasets

- TransformerLens v2.x — github.com/TransformerLensOrg/TransformerLens
- JailbreakBench — jailbreakbench.github.io
- SORRY-Bench — sorry-bench.github.io
- StrongREJECT — the automated refusal evaluator of record
- XSTest — over-refusal benchmark

---

## 10. Out-of-Scope Questions

Scope is Llama-3.2-3B-Instruct specifically. Natural follow-ups, not this project's burden:

- **Does low-rank refusal scale with parameter count?** Would require running the same pipeline on 1B, 3B, 8B, 70B.
- **Is the low-rank structure a training-data artifact or an architectural property?** Requires comparison across model families.
- **Does the harmfulness/refusal disentanglement replicate here?** Zhao tested Gemma and Llama-3.1; adding Llama-3.2-3B is a free bonus contribution from this project.

---

*Revised April 19, 2026 to incorporate post-January field developments: low-rank refusal in small Llamas (IvanC, March 2026), harmfulness/refusal positional disentanglement (Zhao et al., 2025), SAE decomposition work (O'Brien et al., EMNLP 2025), and tooling shifts (TransformerLens v3, nnterp). The January 2026 version is preserved in git history.*
