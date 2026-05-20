# The Refusal Bench: re-scoring six 2025 refusal-ablation techniques on Llama-3.2-1B

> *Draft — May 19 2026. ~1,200 words. Three TBDs flagged inline; the actual numbers from the live bench run and the deployed URLs are the only things blocking publication.*

---

I'm a full-stack engineer at MiQ, teaching myself mechanistic interpretability after hours. The last three months have produced two things worth talking about together: **NeuroScope-Web**, a browser-resident interpretability workbench, and **The Refusal Bench**, a head-to-head comparison of six published refusal-ablation techniques on Llama-3.2-1B-Instruct.

This post is about the second one. The bench is the actual research artifact; the tool is the substrate that made it cheap to build.

## What's actually being measured

Six refusal-ablation techniques published between June 2024 and May 2026 each claim to make a refusal-trained chat model comply with harmful requests by intervening on its internals. Each paper benchmarks against Arditi 2024 — the single-direction baseline — and reports refusal-rate drops via keyword matching or [StrongREJECT](https://arxiv.org/abs/2402.10260)-style scoring.

**No published paper has scored these techniques with a residual-stream harmfulness probe.** That's the gap the bench fills. The headline question:

> When a technique drops the model's *verbal* refusal rate to near zero, does the model's *internal* harmfulness representation also disappear — or does the residual stream still encode "this request is harmful," just without the surface refusal?

That's the Zhao 2507.11878 dissociation, applied across six techniques on the same model, with the same probe, on the same prompts. If a technique scores high on Δ-refusal-rate but low on Δ-AUC, it's a *behavioral patch* — verbal refusal suppressed, harmfulness understanding preserved.

## The six techniques

| Technique | Paper | What it does |
|---|---|---|
| **Arditi (single direction)** | [2406.11717](https://arxiv.org/abs/2406.11717), NeurIPS 2024 | Difference-of-means over (harmful, harmless) prompts, project the resulting unit vector out of the residual stream. |
| **Wollschlager (concept cone)** | [2502.17420](https://arxiv.org/abs/2502.17420), ICML 2025 | Iterative orthogonal extraction. Rank-5 cone of mechanistically distinct refusal directions, all projected out at once. |
| **COSMIC (Luu et al.)** | [2506.00085](https://arxiv.org/abs/2506.00085), ACL 2025 | Auto-selects layer + direction via a cosine-similarity-to-refusal-completion objective. |
| **Cheng (compressed)** | [2604.08524](https://arxiv.org/abs/2604.08524), Apr 2026 | Arditi direction → keep only the top 5% of coefficients by magnitude → re-normalize → ablate. |
| **Maskey (decomposition)** | [2603.27518](https://arxiv.org/abs/2603.27518), Apr 2026 | Separate the over-refusal direction (mild-but-edgy prompts, via [XSTest](https://arxiv.org/abs/2308.01263)) from the harmful direction; ablate only the residual after projecting over-refusal out. |
| **Herring CNA** | [2605.12290](https://arxiv.org/abs/2605.12290), May 2026 | Contrastive Neuron Ablation — identify the ~0.1% of MLP neurons that fire most differentially between harmful and harmless prompts; zero them. |

Five operate on the residual stream as directions; one (Herring CNA) operates on MLP neurons. All six are implemented from their papers as drop-in `Technique` subclasses sharing a common interface (`fit` → `make_ablation_hook`) so the bench can sweep them uniformly.

## The setup

- **Model:** `meta-llama/Llama-3.2-1B-Instruct`, the smallest open Llama that still actually refuses. This is the specific cell of the literature where comparisons haven't been done — Cheng covers 3.2-3B, Herring covers 1B and 3B but with keyword scoring, COSMIC and Maskey use 8B+.
- **Probe:** A logistic-regression harmfulness probe trained on residual stream activations at the ablation layer. AUC measured on held-out (eval) splits.
- **Prompts:** 50 harmful from [JailbreakBench](https://github.com/JailbreakBench/JBB-Behaviors), 50 harmless from Alpaca, length-matched on the Llama-3.2-1B tokenizer. Plus 50 over-refusal prompts from XSTest for the Maskey decomposition step.
- **Layer:** 8 (mid-layer; Wollschlager's preferred for Llama-family models). Single-layer sweep — the techniques each pick their own layer if they prefer (COSMIC does).
- **Split:** 80/20 train (extraction) / eval. Probe trained on extraction; refusal-rate and AUC measured on eval.

## The two-axis result

{{TBD — paste a 6-row table here once the bench has been run on the live HF Spaces backend with Llama-3.2-1B loaded. Each row: technique, layer used, Δ refusal-rate, Δ AUC, elapsed seconds, dissociation flag.}}

```
TECHNIQUE              Δ REFUSAL RATE      Δ AUC          DISSOCIATED?
Arditi (single)        {{TBD}}             {{TBD}}        {{TBD}}
Wollschlager (cone)    {{TBD}}             {{TBD}}        {{TBD}}
COSMIC                 {{TBD}}             {{TBD}}        {{TBD}}
Cheng (compressed)     {{TBD}}             {{TBD}}        {{TBD}}
Maskey (decomp.)       {{TBD}}             {{TBD}}        {{TBD}}
Herring CNA            {{TBD}}             {{TBD}}        {{TBD}}
```

A technique flagged "dissociated" satisfies `|Δ refusal-rate| ≥ 0.3 AND |Δ AUC| ≤ 0.1` — the empirical signature of a behavioral patch. **The headline claim of this post is: {{TBD — fill in based on actual results.}}**

If most or all techniques dissociate, the field's reported "we removed refusal" claims are actually "we removed *the saying of* refusal" claims. If only a subset dissociate, that gives us a ranking — the ones that drop AUC alongside refusal-rate are the genuinely safety-relevant interventions, the others are jailbreaks dressed up in interpretability prose.

## Where this could be wrong

Three caveats that belong in the discussion section, not buried in footnotes:

1. **Single-layer scoring.** The bench picks one layer per technique. Some techniques' effects propagate differently across depth; a layer sweep would be more rigorous and is the obvious follow-up.
2. **Keyword-based refusal rate.** Standard in the field but noisy. Llama-3.2-1B's refusal vocabulary may not match the canonical phrase list, leading to either over- or under-counting. The probe-based AUC measurement doesn't have this problem, but the refusal-rate axis does.
3. **The probe-novelty moat is thinner than it looks.** [Llorente-Saguer 2603.27412](https://arxiv.org/abs/2603.27412) and [Shah 2507.21141](https://arxiv.org/abs/2507.21141) both use residual-stream probes for harmfulness-adjacent classification. The defensible novelty here is the *combination*: a probe-scored *head-to-head* of *six published techniques* on *Llama-3.2-1B-Instruct specifically*. Each of those three qualifiers is necessary.

The closest prior comparable work is Herring 2605.12290 itself — the May 2026 paper that introduced the CNA technique. It tests Llama-3.2-1B, ablates against Arditi, scores with keyword + StrongREJECT. This bench differs in **scoring** (residual-stream probe, which Herring doesn't use) and **breadth** (six techniques, not two).

## Reproduce it yourself

The whole thing is open-source.

- **Code:** {{TBD — repo URL}}
- **Live URL:** {{TBD — Vercel URL}}
- **The bench:** open the URL → load Llama-3.2-1B (you'll need a HuggingFace token with Meta's Llama license accepted) → scroll to Section VII → "↪ run bench" → 6 rows render with your own results in about 90 seconds on a T4 GPU.
- **The data:** `python backend/scripts/build_refusal_pairs.py --n 50 && python backend/scripts/build_over_refusal_pairs.py --n 50` populates `refusal_pairs.py` + `over_refusal_pairs.py` from the public datasets.
- **The techniques:** `backend/refusal_bench/techniques/*.py`. Each ~150–250 lines, each implements the same `Technique(fit, make_ablation_hook)` interface, each cites the paper it replicates in the module docstring.

## What's next

If the dissociation pattern holds on Llama-3.2-1B, the natural extensions are:

- **Llama-3.2-3B as a cross-scale check.** Same techniques, larger same-family model. Does the dissociation persist at 3B?
- **Gemma-2-2B-it as a cross-family check.** Same six techniques, different model family. Does the technique ranking transfer? [Gemma Scope SAEs](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/) would also enable an SAE-grounded version of the probe for comparison.
- **Adversarial robustness.** Re-score after [past-tense reformulation](https://arxiv.org/abs/2407.11969) — does the technique ranking survive prompt perturbation? My guess: no. Which would be its own paper.

If the pattern doesn't hold — if AUC drops alongside refusal-rate consistently — then the field's existing scoring is fine and this whole bench is a calibration exercise. Either way, the number is worth knowing.

If you've worked on refusal-direction interp on small models, or have opinions about the probe-scoring methodology, open an issue or reply on the LessWrong cross-post.

---

*NeuroScope-Web is open-source ({{TBD — repo URL}}). The full landscape brief (May 19 2026) lives in [docs/RESEARCH_LANDSCAPE_2026.md](../docs/RESEARCH_LANDSCAPE_2026.md). For a beginner-up walkthrough of the math, the nine-lesson curriculum lives in [docs/lessons/](../docs/lessons/).*
