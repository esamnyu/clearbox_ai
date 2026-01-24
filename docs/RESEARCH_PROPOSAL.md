# NeuroScope Research Proposal

*Draft — January 2025*
*Status: Awaiting team decisions on open questions*

---

## Context & Origin

This proposal emerged from a consultation with a Google DeepMind researcher who provided feedback on the current NeuroScope project direction. Key insights from that conversation:

1. **GPT-2 cannot be used for refusal research** — it was never RLHF'd to refuse
2. **Experimental rigor requires**: contrastive datasets → activation mapping → ablation verification
3. **Polysemanticity concerns may be overstated** — findings tend to generalize across model sizes
4. **Clean causal claims** (not just visualizations) are what differentiates strong research

---

## Project Type

**This is a learning project, not a publication target.**

| What This Is | What This Isn't |
|--------------|-----------------|
| Demonstrable competence | Novel research contribution |
| Portfolio piece | Paper submission |
| Interview talking point | First-to-publish |
| Skill building | Timeline-pressured |

The goal is to **demonstrate you can execute rigorous methodology**, not to discover something new. Replication or extension of known techniques is perfectly valid.

---

## Team & Goals

| Team Member | Primary Goals | Secondary Goals |
|-------------|---------------|-----------------|
| **Moon** | PhD application differentiator, demonstrate research skills | Potential job at safety-focused lab |
| **Ethan** | Skills/portfolio building, learning | Potentially new job |

**Shared Goal**: Build an interactive tool that demonstrates both research methodology (Moon) and engineering depth (Ethan).

---

## Vision: Real-Time Interpretability Tool

The core idea: **Type a prompt → see the model's internals light up → understand what's happening.**

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuroScope                               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Prompt: "How do I pick a lock?"                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Layer 0      │  │ Layer 1      │  │ Layer 2      │ ...  │
│  │ ┌──┬──┬──┐   │  │ ┌──┬──┬──┐   │  │ ┌──┬──┬──┐   │      │
│  │ │H0│H1│H2│   │  │ │H0│H1│H2│   │  │ │H0│H1│H2│   │      │
│  │ └──┴──┴──┘   │  │ └──┴──┴──┘   │  │ └──┴──┴──┘   │      │
│  │  ■■□ □■□     │  │  □■■ ■□□     │  │  ■■■ □□■     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  [Attention Heatmap]    [Activation Magnitude]    [PCA]    │
│                                                             │
│  Model output: "I can't help with that..."                 │
│                                                             │
│  🔴 Refusal detected    Head 4.7 activated (Δ = 0.83)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Research Question

> **"Can we identify attention heads involved in refusal behavior (or other target behaviors) and verify their causal role through ablation?"**

Framed as learning: We're not claiming novelty—we're demonstrating we can execute the methodology and interpret results.

### Why This Question

- **Methodologically sound**: Follows established contrastive + ablation framework
- **Aligned with safety**: Relevant to PhD programs Moon applied to
- **Verifiable**: Clear success criteria
- **Flexible**: Can pivot to non-refusal behaviors (induction heads, sentiment) if model migration is painful

---

## Methodology

### Phase 1: Dataset Construction
- Collect N prompts that trigger target behavior (e.g., refusal)
- Collect N matched prompts that don't trigger it (benign)
- Ensure semantic similarity except for the target aspect

### Phase 2: Activation Mapping
- Run both sets through model
- Extract attention patterns at every head
- Compute activation difference: `Δ = mean(target) - mean(baseline)`
- Rank heads by `|Δ|` to find candidates

### Phase 3: Ablation Verification
- For top-k candidate heads:
  - Zero out that head's contribution
  - Re-run target prompts
  - Measure: Does the behavior change?
- Report: Which heads, when ablated, change behavior

### Phase 4: Generalization (Optional)
- Test on held-out prompts
- Test across different categories
- Report: Is the finding general or specific?

---

## Architecture

### Current State

```
┌─────────────────────────────────────────────────────┐
│ FRONTEND (React + TypeScript + Vite)                │
│ - Prompt input, model loading UI                    │
│ - Zustand state management                          │
│ - Web Worker with transformers.js                   │
├─────────────────────────────────────────────────────┤
│ PYTHON BACKEND (FastAPI + TransformerLens)          │
│ - /logit-lens, /attention, /gradients               │
│ - /steering-vector, /pca-trajectories               │
│ - Singleton model loading                           │
└─────────────────────────────────────────────────────┘
```

### Visualization Components to Build

| Visualization | Purpose | Backend Support | Frontend Status |
|---------------|---------|-----------------|-----------------|
| Attention heatmap | See where each head attends | `attention.ts` + `/attention` | Not built |
| Head activation grid | Overview of all heads | Extraction exists | Not built |
| Logit lens | Layer-by-layer predictions | `/logit-lens` endpoint | Not built |
| Token importance | Which input tokens matter | `/gradients` endpoint | Not built |
| 3D PCA trajectory | Residual stream through layers | `/pca-trajectories` | Not built |
| Ablation toggle | Disable heads, re-run | Not built | Not built |

---

## Division of Labor

| Component | Owner | Deliverable |
|-----------|-------|-------------|
| Contrastive dataset curation | Moon | Prompt pairs for target behavior |
| Model migration (if needed) | Ethan | Backend running new model |
| Activation extraction API | Ethan | Fast endpoints for per-head data |
| Ablation infrastructure | Ethan | Mechanism to zero out heads |
| Visualization components | Ethan | D3/Canvas/Three.js renderers |
| Statistical analysis | Moon | Metrics, head ranking |
| Interpretation guide | Moon | Help users understand visualizations |
| Write-up (if desired) | Moon (lead) | Blog post or technical report |

---

## Expected Outcomes

### For Moon
- Demonstrated experimental design skills
- Something concrete to discuss in PhD interviews
- Understanding of mechanistic interpretability techniques
- Portfolio piece showing research + engineering collaboration

**Interview story**: "We built an interpretability toolkit and used it to investigate [behavior] in [model]. Here's what we found, here's what surprised us, here's what we'd do differently."

### For Ethan
- Full-stack ML engineering experience
- Visualization portfolio (D3, WebGL, real-time data)
- TransformerLens / activation extraction proficiency
- Collaboration experience with a researcher

---

## Open Questions

### Q1: Model Choice

**Context**: GPT-2 was never RLHF'd, so it can't refuse. To study refusal, you need a model that actually refuses.

| Option | Pros | Cons |
|--------|------|------|
| **A) Stay on GPT-2** | Already working, mature TransformerLens support | Cannot study refusal; limited to induction heads, sentiment, etc. |
| **B) Migrate to Gemma-2-2B** | RLHF'd, can study refusal, Google's open model | Community TransformerLens support, may have issues |
| **C) Migrate to Llama-3.2-1B/3B** | RLHF'd, popular, active community | Requires TransformerLens config, Meta's model |
| **D) Test both, decide later** | Informed decision | Delays starting |

**Decision**: _______________

**Action required before committing**:
- [ ] Test loading Gemma-2-2B in TransformerLens
- [ ] Test loading Llama-3.2-1B in TransformerLens
- [ ] Verify activation extraction works
- [ ] Verify model actually refuses harmful prompts

---

### Q2: Target Behavior to Study

| Option | Description | Model Requirement |
|--------|-------------|-------------------|
| **A) Refusal** | Find heads that control refusal responses | Requires RLHF'd model |
| **B) Induction heads** | Replicate known findings about pattern-copying | GPT-2 works fine |
| **C) Sentiment** | Find heads that encode positive/negative sentiment | GPT-2 works fine |
| **D) Factual recall** | Heads involved in retrieving facts | GPT-2 works fine |
| **E) Multiple** | Start with one, expand to others | Depends on choice |

**Decision**: _______________

---

### Q3: Scope of Dataset (if studying refusal)

| Option | Description | Effort |
|--------|-------------|--------|
| **A) Single category** | Focus on one type (e.g., violence) | Lower, cleaner experiment |
| **B) Multiple categories** | Cover 3-4 refusal types | Medium, stronger claim |
| **C) Universal** | Try to find a single "refusal head" across all categories | Higher, riskier |

**Decision**: _______________

---

### Q4: Level of Real-Time

| Option | Description | Engineering Effort |
|--------|-------------|-------------------|
| **A) On-submit** | Full results after generation completes (~1-2 sec) | Moderate |
| **B) Streaming** | Token-by-token activation updates | High |

**Recommendation**: Start with on-submit; streaming is a nice-to-have.

**Decision**: _______________

---

### Q5: Which Visualizations First?

Pick 1-2 to start (can add more later):

- [ ] **Attention heatmap** — Most common, well-understood
- [ ] **Head activation grid** — Overview, good for "which head matters"
- [ ] **Logit lens** — Layer-by-layer predictions, shows model's "thinking"
- [ ] **Token importance** — Gradient-based highlighting of input
- [ ] **3D PCA trajectory** — Flashy, but harder to interpret

**Decision**: _______________

---

### Q6: How Deep Do You Want to Go?

| Level | Stopping Point | What You Can Demonstrate |
|-------|----------------|--------------------------|
| **A) Pipeline only** | Build extraction + visualization infra | "We built the tooling" |
| **B) One experiment** | Full methodology on one behavior | "We investigated X and found Y" |
| **C) Comparative** | Multiple behaviors or models | "We compared across X and Y" |

**Decision**: _______________

---

### Q7: What Happens to Existing GPT-2 Work?

| Option | Description |
|--------|-------------|
| **A) Archive it** | Keep as learning artifact, don't continue |
| **B) Parallel track** | Maintain GPT-2 for non-refusal work, new model for refusal |
| **C) Port patterns** | Use GPT-2 codebase as template, fully migrate |

**Decision**: _______________

---

### Q8: Where Does It Run?

| Option | Pros | Cons |
|--------|------|------|
| **A) Browser only** (transformers.js) | No server needed, portable | Slower, limited models |
| **B) Python backend** | Faster, full TransformerLens | Requires running server |
| **C) Hybrid** (current) | Best of both | More complexity |

**Decision**: _______________

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TransformerLens doesn't support chosen model | Medium | High | Test early; have backup model/behavior |
| No clear "behavior head" exists | Medium | Medium | Still valuable learning; report what you found |
| Ablation doesn't change behavior | Medium | Medium | Negative results are still valid if methodology is sound |
| Visualization takes longer than expected | Medium | Low | Start with simple viz, iterate |
| Scope creep | High | Medium | Decide on scope level (Q6) and stick to it |

---

## Next Steps

1. **Moon + Ethan**: Review this proposal and the [Research Strategy](./RESEARCH_STRATEGY.md) document
2. **Ethan**: Run model validation tests (TransformerLens + target behavior)
3. **Moon**: Begin curating prompt pairs for target behavior
4. **Reconvene**: Finalize decisions, assign first tasks

> **📄 See Also:** [RESEARCH_STRATEGY.md](./RESEARCH_STRATEGY.md) — Deep research analysis with recommended answers to the open questions above, methodology details, dataset guidance, and a 4-week implementation roadmap.

---

## References

- Consultation transcript (January 2025) — GDM researcher feedback on experimental design
- Olsson et al. — "In-context Learning and Induction Heads"
- Elhage et al. — "Toy Models of Superposition"
- TransformerLens documentation — https://neelnanda-io.github.io/TransformerLens/

---

## Appendix: Skills Checklist

Use this to clarify what each person wants to learn:

**Moon**:
- [ ] Experimental design for interpretability
- [ ] TransformerLens proficiency
- [ ] Ablation/causal analysis techniques
- [ ] Something concrete for PhD interviews
- [ ] Other: _______________

**Ethan**:
- [ ] ML model serving infrastructure
- [ ] Activation extraction pipelines
- [ ] Frontend visualization (D3, WebGL, Canvas)
- [ ] Testing strategies for ML systems
- [ ] Full-stack ML engineering
- [ ] Other: _______________
