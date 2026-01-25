# NeuroScope Research Strategy

> Deep Research Analysis for Mechanistic Interpretability Project
> January 2026

---

## Executive Summary

Based on consultation with Google DeepMind research guidance and comprehensive literature review, this document provides strategic recommendations for the NeuroScope project. The core insight: **refusal behavior in LLMs is mediated by a single direction in activation space**, making it an excellent target for demonstrating rigorous interpretability methodology.

### Key Recommendations

1. **Model**: Llama-3.2-3B-Instruct (best TransformerLens support, manageable size)
2. **Target Behavior**: Refusal mechanisms (well-documented methodology exists)
3. **Methodology**: Difference-of-means with directional ablation verification
4. **Dataset**: JailbreakBench (100 harmful) + Alpaca subset (100 harmless)
5. **Scope**: One complete experiment with full methodology (Level B)

---

## 1. Model Choice Decision

### The Core Problem

GPT-2 was never RLHF-trained, so it cannot refuse harmful requests. To study refusal mechanisms, you need a model that actually refuses. The DeepMind researcher was correct: this is a fundamental constraint.

### TransformerLens Compatibility Analysis

| Model | Support Status | VRAM Needed | Known Issues |
|-------|---------------|-------------|--------------|
| Llama-3.2-1B-Instruct | Fully Supported | ~4GB | None significant |
| **Llama-3.2-3B-Instruct** | **Fully Supported** | **~8GB** | **None significant - RECOMMENDED** |
| Gemma-2-2B-it | Supported | ~8GB+ | High VRAM usage bug reported |
| Qwen-2.5-3B-Instruct | Supported | ~8GB | Rotary base config needed |

### Recommendation: Llama-3.2-3B-Instruct

- Best TransformerLens support with no known issues
- Small enough to run on consumer GPU (8GB VRAM)
- RLHF+DPO trained, so it reliably refuses harmful prompts
- Used in the foundational "Refusal Direction" paper (Arditi et al., NeurIPS 2024)
- Active community support and existing abliteration research

---

## 2. Research Methodology

### The Refusal Direction Method (Arditi et al., NeurIPS 2024)

This is the gold standard methodology for studying refusal. It demonstrates that refusal behavior is encoded in a one-dimensional subspace, making it tractable for analysis and verification.

### Phase 1: Data Collection

1. Run model on n harmful instructions (n=100-512)
2. Run model on n harmless instructions (matched count)
3. Cache residual stream activations at last token position
4. Focus on upper layers (e.g., layer 10-15 for Llama-3.2)

### Phase 2: Compute Refusal Direction

For each layer, compute:

> *r = mean(harmful_activations) - mean(harmless_activations)*

Normalize this vector. The direction r points from "compliant" to "refusing" in activation space.

### Phase 3: Causal Verification

**Necessity Test (Ablation):**
- Remove the refusal direction component from activations
- Re-run harmful prompts
- Expected: Model no longer refuses (attack success rate increases)

**Sufficiency Test (Addition):**
- Add the refusal direction to activations on harmless prompts
- Expected: Model starts refusing harmless requests

---

## 3. Dataset Construction

### Recommended Datasets

| Dataset | Size | Use Case |
|---------|------|----------|
| **JailbreakBench** | 100 behaviors | Primary harmful prompt source - curated, aligned with OpenAI policies |
| AdvBench | 520 examples | Extended harmful behaviors including profanity, threats, misinformation |
| Alpaca | 52,000 | Harmless instruction-following (sample 100-500 for balance) |
| XSTest | 450 prompts | Over-refusal evaluation (false positives / exaggerated safety) |

### Sample Size Guidelines

| Purpose | Minimum | Recommended |
|---------|---------|-------------|
| Initial exploration | 32 pairs | 64-100 pairs |
| Direction extraction | 100 pairs | 200-500 pairs |
| Held-out validation | 50 pairs | 100+ pairs |

### Dataset Quality Checklist

- **Single feature isolation:** Pairs differ only in target aspect (harmful vs harmless)
- **Token length matching:** Similar token counts prevent positional artifacts
- **Semantic similarity:** Same structure/style, only harm-related content differs
- **Chat template compliance:** Use model's instruction format consistently

---

## 4. Visualization Strategy

Start with the most informative visualizations that directly support the research question.

### Priority 1: Head Activation Grid

Shows which heads are most active when the model refuses. This directly answers "which components matter?" and helps identify candidate refusal heads for ablation.

### Priority 2: Attention Heatmap

Shows where each head attends. Once you identify candidate refusal heads from the grid, attention patterns reveal what those heads are "looking at" when they activate.

### Priority 3: Logit Lens

Layer-by-layer predictions showing how the model's "thinking" evolves. Useful for seeing when refusal emerges in the forward pass.

### Defer: 3D PCA Trajectory

While visually compelling, PCA trajectories are harder to interpret and don't directly support causal claims. Implement after core analysis is complete.

---

## 5. Common Pitfalls to Avoid

### Methodological Pitfalls

1. **Zero Ablation:** Use mean ablation instead. Zero has no special meaning in activation space.

2. **Single Direction Overconfidence:** Recent work shows refusal may be multidimensional. Report cosine similarities between topic-specific directions.

3. **Ignoring Collateral Damage:** Ablation can affect model capabilities. Always test on harmless prompts too.

4. **Insufficient Validation:** Test on held-out sets AND adversarial jailbreak prompts.

### Conceptual Pitfalls

1. **Correlation vs Causation:** Always perform intervention experiments (ablation + addition), not just correlation analysis.

2. **Semantic Overinterpretation:** The refusal direction's semantic meaning is unclear. Report what you observe, not what you assume.

3. **Generalization Assumptions:** Findings on one model may not transfer. Be explicit about scope.

---

## 6. Answers to Open Questions

| Question | Recommendation | Rationale |
|----------|---------------|-----------|
| Q1: Model Choice | **Llama-3.2-3B-Instruct** | Best TransformerLens support, proven for refusal research |
| Q2: Target Behavior | **Refusal mechanisms** | Well-documented methodology, aligns with safety relevance for Moon |
| Q3: Dataset Scope | **Multiple categories (3-4 types)** | Stronger claim, test generalization across harm types |
| Q4: Real-Time Level | **On-submit (not streaming)** | Lower complexity, sufficient for research goals |
| Q5: First Visualizations | **Head activation grid + Attention heatmap** | Directly support identifying and analyzing refusal components |
| Q6: Depth Level | **Level B: One complete experiment** | Demonstrates full methodology without overcommitting |
| Q7: GPT-2 Work | **Option C: Port patterns to new model** | Use existing codebase as template, don't maintain two tracks |
| Q8: Runtime Environment | **Python backend (TransformerLens)** | Required for proper activation extraction and ablation |

---

## 7. Recommended Implementation Roadmap

### Week 1: Foundation

- **Ethan:** Verify Llama-3.2-3B-Instruct loads in TransformerLens, test activation extraction
- **Moon:** Download JailbreakBench, sample Alpaca, verify model refuses harmful prompts

### Week 2: Direction Extraction

- **Ethan:** Build activation caching pipeline, expose via API endpoint
- **Moon:** Compute difference-of-means across layers, identify best layer for refusal direction

### Week 3: Causal Verification

- **Ethan:** Implement ablation infrastructure (direction removal hooks)
- **Moon:** Run ablation experiments, measure attack success rate changes

### Week 4: Visualization & Write-up

- **Ethan:** Build head activation grid and attention heatmap visualizations
- **Moon:** Interpret results, draft findings for portfolio/interview story

---

## 8. Key References

### Foundational Papers

- **Arditi et al., NeurIPS 2024:** "Refusal in Language Models Is Mediated by a Single Direction" - Primary methodology
- **Heimersheim, 2024:** "How to Use and Interpret Activation Patching" - Best practices for intervention experiments
- **Open Problems in Mechanistic Interpretability, Jan 2025:** Current research landscape overview from FAR.AI

### Tools & Datasets

- **TransformerLens:** github.com/TransformerLensOrg/TransformerLens
- **JailbreakBench:** jailbreakbench.github.io
- **Abliteration Tutorial:** huggingface.co/blog/mlabonne/abliteration

---

*This document was compiled from deep research across academic papers, GitHub repositories, and interpretability community resources. The recommendations prioritize demonstrable competence and methodological rigor over novelty, aligning with the project's goals as a learning exercise and portfolio piece.*
