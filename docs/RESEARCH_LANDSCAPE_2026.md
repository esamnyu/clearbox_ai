# NeuroScope Research Landscape — April 2026

> Strategic refresh of `RESEARCH_PROPOSAL.md` and `RESEARCH_STRATEGY.md` (both Jan 2026).
> Purpose: decide what to actually build over the next 4 weeks given how the field has moved.
> Audience: Ethan, now driving both engineering and research direction.

---

## TL;DR

1. **Arditi et al.'s "single direction" claim is now a baseline, not the ceiling.** Three follow-ups (Wollschlager ICML 2025, Zhao NeurIPS 2025, community replications March 2026) show the story is richer — and **Llama-family models specifically sit in the harder "low-rank / multi-direction" regime**. A straight Arditi replication on Llama-3.2-3B in April 2026 looks dated.

2. **SAEs have cooled.** Kantamneni et al. (Feb 2025) showed SAE probes underperform plain logistic regression across 113 datasets. DeepMind publicly deprioritized. The 2026 consensus: "SAEs for discovery, not for action on known concepts." Skip standalone SAE probing unless tied to circuit tracing.

3. **Circuit tracing / attribution graphs is the dominant paradigm now.** Anthropic open-sourced their tracer; Gemma Scope 2 shipped Dec 2025; Neuronpedia hosts ~7k user-generated graphs. Incorporating even a partial attribution graph into NeuroScope is a high-signal addition.

4. **You cannot out-scale Neuronpedia.** The defensible niche for a 2-person portfolio project is narrow: **browser-native zero-install workbench** + **one integrated methodology loop** + **one genuine research finding, blogged**. Everything else has stronger competitors.

5. **The current code is behind the strategy doc.** `backend/research.py` and `backend/main.py` are still GPT-2 only (sentiment steering, layer bounds `le=11`). The Llama migration and ablation infrastructure described in `RESEARCH_STRATEGY.md` are not yet implemented. This is actually good news — you can course-correct before shipping migration code for a plan that needs updating.

6. **MATS Summer 2026 applications close April 26 (9 days).** Anthropic Fellows deadline same day. If this project is a PhD/fellowship input, timing pressure is real.

---

## Update — April 28 2026 (11 days later)

This doc was compiled on April 17. In 11 days the field moved enough to need explicit corrections; the ordering and target-model choice in this update **supersede §4 week-by-week and §6 references** where they conflict. Original content preserved below for context.

### Citation corrections (apply to §1 + §6)

- **Zhao 2507.11878** — venue claim "NeurIPS 2025" is **unconfirmed** (no OpenReview record for this title); treat as preprint until verified.
- **Luu/COSMIC 2506.00085** — **ACL 2025 Findings**, not main proceedings.
- **LessWrong (March 2026, IvanC)** — Llama-3.2-3B reaches **~15% compliance with one vector, ~37% with a 5-layer stack**. The "~36% with a single vector" number in the §1 table is wrong; flip those.

### New papers since April 17 affecting predictions

| arXiv | Date | Effect |
| --- | --- | --- |
| **2604.08524** Cheng et al., "What Drives Representation Steering?" | Apr 9 2026 | Direct steering analysis on **Llama-3.2-3B**; vectors compress to 1–10% of dims while preserving effect; isolates an OV-circuit pathway. **Strongest scoop on the original prediction 1.** |
| **2603.27518** Maskey et al., "Over-Refusal and Representation Subspaces" | rev Apr 22 2026 | Reframes rank > 1 as **over-refusal vs harmful-refusal** — task-conditioned, not "safety-implementation depth." **Reframes the original prediction 1.** |
| **2602.02132** Joad et al., "There Is More to Refusal than a Single Direction" | Feb 2 2026 | Per-category geometrically distinct directions that collapse to one knob. Partially scoops the framing. |
| **2603.13359** Alagharu et al., "Refusal Tokens to Refusal Control" | Mar 9 2026 | Category-specific low-rank directions on Llama-3-8B. |
| **2601.16034** Cristofano, "Universal Refusal Circuits" | Jan 2026 | Cross-model low-dim universal refusal structure. |
| 2604.03147 Sun et al. — Valence-Arousal subspace | Apr 3 2026 | Tangential (8B+ only). |
| 2604.11120 Li et al. — "Persona Non Grata" | Apr 13 2026 | Tangential (3.1-8B). |

No public tool reproduces the exact compliance-vs-rank-k curve on Llama-3.2-3B yet, but the easy version of that experiment is now half-scooped by LessWrong + Cheng.

### Revised prediction ordering (supersedes §4 below)

The April 17 plan led with the rank-k structure. April moves push that to "replication," not "headline finding." New order:

1. **(LEAD)** Harmfulness probe after refusal-direction ablation on **Llama-3.2-1B / 3B** — replicates Zhao (preprint) which tested 8B+ models. Nobody has run this on the small Llamas. **Still open. This is the headline.**
2. **(Replication + extension)** Compliance-vs-rank-k curve on Llama-3.2-1B / 3B — frame as confirmation, not discovery. Cite Wollschlager + LessWrong + Cheng 2604.08524 + Maskey 2603.27518 as state-of-the-art.
3. **(Defer or drop)** Subspace-rerouting jailbreak evasion — overlaps Winninger 2503.06269. Move to "future work."

### Target model — flip primary to Llama-3.2-1B

Two reasons to switch from 3B → 1B as primary, with 3B as confirmation:

- **Already partially scooped on 3B.** LessWrong + Cheng 2604.08524 cover 3B; 1B is genuinely under-explored.
- **Engineering thesis fit.** 1B is plausibly browser-loadable (≈2.5 GB int8); 3B is not. Picking 1B *strengthens* the "browser-native zero-install workbench" claim instead of contradicting it.
- Same low-rank refusal phenomenon should hold (same family, same architecture).

### Codebase state vs strategy (April 28)

| Status | Item |
| --- | --- |
| ✅ committed (`21457c8`) | Deploy infrastructure: Dockerfile, env-driven CORS, env-driven API base, pinned requirements, `docs/DEPLOYMENT.md` |
| ✅ uncommitted | **Ablation primitive built**: `backend/research.py::ablate_along_direction`, `POST /ablate-direction`, frontend client, store, UI section in `SteeringPanel.tsx` |
| ⏳ | Llama-3.2-1B migration — out of scope for the immediate window; layer bound `le=11` in `backend/main.py` will need to become `le=15` for 1B (16 layers) and `le=27` for 3B (28 layers) |
| ⏳ | Live deploy: Dockerfile + env wiring done; HF Space + Vercel project not yet created |
| ⏳ | Blog post + Loom + filled-in research statement |

---

## 1. What's Changed Since Your Jan 2026 Strategy Doc

### The refusal-direction story is now multi-paper

| Paper | Finding | Why it matters for you |
|---|---|---|
| [Wollschlager et al. ICML 2025](https://arxiv.org/abs/2502.17420) | Refusal lives in a **polyhedral "concept cone"** of mechanistically independent directions, not a single line. Introduces "representational independence" — orthogonal ≠ causally independent under intervention. | The paper you must cite. Either build on it or argue against it. |
| [Zhao et al. NeurIPS 2025](https://arxiv.org/abs/2507.11878) | **Harmfulness and refusal are distinct directions.** Ablating Arditi's refusal direction suppresses the verbal refusal but does NOT remove the model's internal harmfulness judgment. | Cheap add-on experiment. If you claim you've "disabled refusal," Zhao will be cited against you. Probe both directions and report separately. |
| [Luu et al. ACL 2025 — COSMIC](https://arxiv.org/abs/2506.00085) | Automated direction/layer selection via cosine-similarity objective — no refusal-token heuristic required. | Run as a sanity check alongside Arditi; costs ~1 day of compute; demonstrates methodological rigor. |
| [Prakash et al. Sept 2025](https://arxiv.org/abs/2509.09708) | Three-stage SAE pipeline (refusal direction → greedy feature filter → factorization machine) on Gemma-2-2B-IT and Llama-3.1-8B-IT. | Closest methodological upgrade for your target model family. |
| [LessWrong, March 2026](https://www.lesswrong.com/posts/LMkvjDTLKFrgdzJdG/single-direction-vs-low-rank-refusal-in-small-llms-1) | Qwen is ~91% compliance with a single vector. **Llama-family is low-rank/multi-direction — single vector only ~36% compliance.** | **Direct hit on your chosen model.** Plan to report a rank-k basis for Llama-3.2-3B, not a single vector. |
| [Wang et al. NeurIPS 2025](https://arxiv.org/abs/2505.17306) | English-derived refusal direction transfers across 14 languages. | Cross-lingual is already done. Cross-architecture within English is still open. |

**Upshot:** a 1D-direction-only writeup on Llama-3.2-3B in April 2026 looks uninformed. The field has moved to multi-direction geometry and harmfulness/refusal separation.

### SAEs: genuine cooling, not hype

- [Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda (Feb 2025)](https://arxiv.org/abs/2502.16681) — SAE probes underperform plain logistic regression baselines on 113 probing/steering datasets.
- DeepMind's mech interp team publicly repositioned. Nanda's current framing: moved from "low chance of incredibly big deal" to "high chance of medium big deal." The phrase is **"pragmatic interpretability."**
- Position response: ["Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts"](https://arxiv.org/html/2506.23845v1) — the intellectually honest repositioning.
- Architectures still active: [JumpReLU](https://arxiv.org/html/2407.14435v1), [Transcoders](https://arxiv.org/html/2501.18823v1) (beat vanilla SAEs), Matryoshka SAEs (default in Gemma Scope 2), Switch SAEs.
- **No first-party pretrained SAE exists for Llama-3.2-3B-Instruct** as of April 2026. [PaulPauls/llama3_interpretability_sae](https://github.com/PaulPauls/llama3_interpretability_sae) has a layer-23 SAE; [FAST (2506.07691)](https://arxiv.org/html/2506.07691v1) tuned some for 3.2-3B-Instruct. Transfer from 3.1-8B SAEs does NOT work (incompatible dims).
- **If you want SAEs: switch target model to Gemma-2-2B where Gemma Scope is turnkey.**

### Circuit tracing / attribution graphs is the new center of gravity

- [Anthropic — Circuit Tracing: Revealing Computational Graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) + companion [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html), March 2025. Cross-Layer Transcoders + attribution graphs on Claude 3.5 Haiku. Applied to planning, arithmetic, hallucination, jailbreaks, refusal.
- [Open-source circuit tracer](https://www.anthropic.com/research/open-source-circuit-tracing) (Anthropic + Neuronpedia, mid-2025). Runs on open-weights models **including Llama**. Frontend hosted on Neuronpedia (~7k user-generated graphs).
- [Gemma Scope 2 (DeepMind, Dec 2025)](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/) — SAEs + transcoders + cross-layer transcoders on all Gemma 3 sizes. 110 PB of activations, >1T SAE parameters.
- **Honest caveat:** Anthropic's own reporting says attribution graphs yield satisfying insight on **~25% of prompts** — and even those capture only part of the mechanism. [ICLR 2026 OpenReview 5IWJBStfU7](https://openreview.net/forum?id=5IWJBStfU7) gives formal NP-hardness / inapproximability results for circuit-finding queries.

### Activation patching — unified, and RelP > AtP

- [Geiger et al., Causal Abstraction (JMLR 2025)](https://jmlr.org/papers/v26/23-0058.html) — unifies activation patching, path patching, causal mediation, causal scrubbing, DAS, concept erasure, SAEs under one causal-abstraction framework. The theory paper to know.
- **Relevance Patching (RelP)** dramatically outperforms Attribution Patching (AtP) for MLP components in deeper networks (GPT-2 Large: AtP 0.006 vs RelP 0.956 correlation with ground truth). AtP is no longer the default.

### Hiring-signal shift

- **Anthropic Interpretability + Alignment Science** — [Fellows 2026](https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/) (May + July cohorts, apps due **April 26, 2026** — 9 days from today). Hires on attribution-graph / circuit-tracing work, empirical rigor, shipped tooling. Their public hiring page explicitly calls out engineering depth as a constraint.
- **DeepMind (Neel Nanda / Arthur Conmy)** — pragmatic shift; deprioritized SAE probing post-Kantamneni; hires work that addresses a concrete safety-relevant task (deception detection, CoT faithfulness, eval-gaming).
- **Apollo Research** — deception/scheming + interp. Dual skillset (evals + interp) matters.
- **MATS** — [Summer 2026](https://www.matsprogram.org/program/summer-2026) is largest cohort ever (120 fellows, 100 mentors). Anthropic + OpenAI "Megastream" is the prestige track. Apps due **April 26, 2026**. Conmy and Sharkey run the mech-interp streams. MATS → PhD is the most reliable current pipeline.
- **Strong PhD producers in mech interp:** Berkeley CHAI, MIT (Tegmark — produced Kantamneni), Oxford (Barez, Mayne), **Northeastern (David Bau — produced Zhao's harmfulness/refusal paper)**, Stanford (Geiger — causal abstraction), CMU (growing), NYU.

**What differentiates applicants in April 2026:**
1. **One real causal intervention,** not just probe/steer.
2. **Engineering visible in a public repo** — reviewers check GitHub for tests, CI, architecture quality.
3. **One narrow, surprising empirical finding** beats a broad survey.
4. **Explicit awareness of SAE skepticism** — pitching SAEs without acknowledging Kantamneni looks out of date.

---

## 2. The Tooling Landscape — What You're Competing With

| Tool | What it is | Why you can't beat it | Anything missing? |
|---|---|---|---|
| [Neuronpedia](https://www.neuronpedia.org/) | Dominant public interpretability platform. 5+ TB of activations, autointerp, attribution-graph UI for Anthropic's tracer. ~800 GitHub stars, fully open-source since 2025. | Scale, infrastructure, model coverage. | It's a catalog more than a workbench. "Live loop" UX (run → inspect → intervene → rerun end-to-end) is weak. Extensibility admitted hard even by maintainers. |
| [TransformerLens + CircuitsVis](https://github.com/TransformerLensOrg/TransformerLens) | ~3.3k stars. De facto research substrate; every serious mech-interp project builds on it. | Canonical library, notebook-native community. | Notebook-bound, atomistic (one chart per call), no unified workbench. |
| [circuit-tracer (Anthropic)](https://github.com/decoderesearch/circuit-tracer) | Attribution-graph generator. ~2.7k stars. Frontend hosted on Neuronpedia. | Official Anthropic-backed tool. | Graph-only; no steering, no contrastive-pair workflow. |
| [Gemma Scope 2 (DeepMind)](https://deepmind.google/models/gemma/gemma-scope/) | SAEs + transcoders + CLTs for all Gemma 3 sizes. Demo on Neuronpedia. | Scale unmatchable. | No standalone UI — DeepMind leverages Neuronpedia. |
| [SAELens](https://github.com/decoderesearch/SAELens) | Canonical SAE training/loading library. | Standard tool. | Not a UI. |
| [nnsight / NDIF (Bau Lab)](https://nnsight.net/) | Remote execution on frontier models via shared GPU fabric. v0.6 Feb 2026. | Frontier-scale access. | No built-in viz layer. |
| [Tuned Lens](https://github.com/AlignmentResearch/tuned-lens) | Canonical "better logit lens." | Reference implementation. | Library, not product. |
| [LLM Transparency Tool (Meta)](https://github.com/facebookresearch/llm-transparency-tool) | Streamlit-based info-flow graphs, attention/contribution maps, logit lens. | Activity slowed. CC BY-NC. | Good reference design — not a living product. |
| [Transformer Explainer (Poloclub)](https://poloclub.github.io/transformer-explainer/) | 490k+ users, in-browser GPT-2, CHI 2026 paper. | Pedagogy gold standard. | Teaching demo, not a research workbench. |
| [EasySteer / EasyEdit2](https://github.com/ZJU-REAL/EasySteer) | Browser UIs for steering / model-editing experiments. | Overlaps your planned steering scope. | Narrow. |
| [InTraVisTo](https://arxiv.org/html/2507.13858v1) | Academic hidden-state inspection + info-flow viz + interactive injection. | Single-paper demo; low adoption. | — |

### Cross-cutting researcher complaints (useful gap list)

- **Tool siloing.** Every tool owns one slice (SAEs, circuits, lenses, steering). Switching means re-tokenizing, re-running, re-exporting.
- **No "live loop" UX.** Run → inspect → intervene → rerun is still mostly notebook cells. Neuronpedia has live feature testing, not end-to-end workflows.
- **GPU / server dependency.** Most tools need an inference backend. Dead ends for browsers, classrooms, rapid prototyping.
- **Shallow coverage of the "early pipeline"** (tokenization, prompt construction) and the **"late pipeline"** (writeup, reproducibility artifacts).
- **Pedagogical tools ≠ research tools.** Transformer Explainer is great for the first hour and useless for hour two.

### The defensible niches for a 2-person project (ranked)

1. **Browser-native, zero-install mech-interp workbench (strongest).** No existing product does end-to-end interp fully client-side. Transformer Explainer is pedagogy-only; WebLLM runs inference but has no interp layer; Neuronpedia is server-bound. `transformers.js` + WebGPU + hook-equivalent instrumentation in the browser is a genuine technical gap. Interview framing: *"serverless, shareable, privacy-preserving interp; one URL = reproducible investigation."* Maps directly onto the engineering signal Anthropic hiring wants.
2. **Integrated methodology loop.** Prompt-pair builder → diff-of-means / contrastive extraction → ablation + steering slider → auto-generated markdown writeup. A *methodology artifact*, not just a visualizer.
3. **Specific-model-deep.** Best possible tool for one model (Llama-3.2-1B or Gemma-2-2B), with canonical probes and known circuits pre-loaded.
4. **Adversarial angle.** Built-in GCG-style search against a steered/ablated model. Upside is high; risk is that browser-tier models make demos feel toy.
5. **Pedagogy angle (weakest).** Transformer Explainer already owns this. Don't lead here.

---

## 3. Where the Research Could Pivot

Three angles that would convert "yet another Arditi replication" into something you can actually pitch. Ranked by leverage.

### Pivot A — Failure-mode audit of Arditi on Llama-3.2-3B (recommended)

Replicate Arditi as Week 1 foundation, then spend Weeks 2–4 characterizing where the single-direction method breaks. Three tractable failure regimes to survey:

- **Rank-k structure** (Wollschlager, LessWrong March 2026): how many orthogonal directions does Llama-3.2-3B need before ablation reliably flips refusal? Report the compliance-vs-rank curve.
- **Harmfulness/refusal separation** (Zhao NeurIPS 2025): after ablating the refusal direction, does a probe on the harmfulness direction still detect harm? If yes, the "refusal" ablation is a lexical patch, not a safety patch.
- **Adversarial evasion**: jailbreak prompts whose hidden states mimic benign instructions. Does the refusal direction detect these? [COSMIC](https://arxiv.org/abs/2506.00085) and [Subspace Rerouting (2503.06269)](https://arxiv.org/pdf/2503.06269) document the failure.

**Interview story:** *"We replicated Arditi on Llama-3.2-3B, found it's not single-direction, mapped out when the simple method breaks, and the tool lets you reproduce any of it in three clicks."* Directly answers the "narrow surprising empirical finding" hiring bar.

### Pivot B — Persona Vectors on Llama-3.2-3B + geometric comparison to refusal

[Chen et al. (Anthropic, arXiv 2507.21509), Sept 2025](https://arxiv.org/abs/2507.21509). Contrastive prompting extracts direction-of-means vectors for 7 traits: evil, sycophancy, hallucination, optimistic, impolite, apathetic, humorous. **Anthropic released the code.**

- Apply the pipeline to Llama-3.2-3B.
- Measure cosine similarity between persona vectors and the refusal direction.
- Ask: is refusal geometrically special, or just one persona axis among many?

High differentiation value; moderate risk because Anthropic published code → many will replicate. Novel angle needed (composition, orthogonality-to-refusal, cross-model transfer).

### Pivot C — Entity hallucination directions (Ferrando et al.)

[Ferrando et al. ICLR 2025 — "Do I Know This Entity?"](https://arxiv.org/abs/2411.14257). SAE directions causally control whether a model refuses vs. hallucinates about known/unknown entities. Chat-tuning repurposes base model machinery.

- Extract entity-recognition directions on base Llama-3.2-3B vs. Instruct.
- Demonstrate the "chat tuning repurposes base mechanism" narrative with your own data.
- Under-replicated; clean methodology; hallucination is a hot 2026 topic.

### Pivots to avoid

- **Pure induction-head replication** (stale).
- **SAE feature steering on Llama-3.2-3B** (SAE infrastructure is fragmented; switch to Gemma-2-2B if SAEs become the focus).
- **Broad cross-lingual refusal generalization** ([Wang et al. NeurIPS 2025](https://arxiv.org/abs/2505.17306) already did the 14-language case).

---

## 4. Recommended Plan (supersedes `RESEARCH_STRATEGY.md` §7 roadmap)

### Research framing

> **"Refusal in Llama-3.2-3B is multi-directional: replicating Arditi and mapping where the single-direction story breaks."**

Deliverables:
1. **A LessWrong / Alignment Forum post** with the empirical finding. This is the portfolio artifact. Hiring reviewers read blog posts more than tools.
2. **A live deployed browser-native tool** that reproduces any result in the post in 3 clicks. This is the engineering artifact.
3. **A clean public GitHub repo** with tests, CI, and architecture notes.

### Week-by-week (4 weeks, solo-driver assumption)

> Reordered April 28 to put the still-open headline finding first. See "Update — April 28 2026" near the top of this doc for context.

| Week | Research | Tooling |
| --- | --- | --- |
| **1** | Replicate Arditi on **Llama-3.2-1B** (primary) with 3.2-3B as confirmation. Extract refusal direction. Verify it flips refusal. Record baseline numbers. | Migrate backend from GPT-2 to Llama-3.2-1B-Instruct. **Ablation hook already built** (`backend/research.py::ablate_along_direction`); generalize the layer bound (currently `le=11`) for Llama. Port to TransformerLens's Llama config. |
| **2** | **(HEADLINE) Harmfulness/refusal separation** following Zhao 2507.11878 — probe the harmfulness direction *before and after* refusal-direction ablation. Zhao tested 8B+; nobody has run this on small Llama. Show the "verbal refusal suppressed but harm still detected" result if it replicates. | Add "harmfulness probe" panel that hooks into the existing ablation flow. Add per-layer extraction view. |
| **3** | **(Confirmation) Compliance-vs-rank-k curve** — iterative / orthogonal ablation following Wollschlager + Maskey 2603.27518. Compare to LessWrong (March 2026) and Cheng 2604.08524. Run COSMIC as a methodology sanity check. | Add multi-direction ablation slider to UI (rank k = 1..10). |
| **4** | Auto-writeup. Live deploy. Blog post draft. (Subspace-rerouting jailbreak evasion → "future work" section.) | Auto-generated-markdown writeup feature. Final deploy polish. Integrate a partial attribution-graph view using `circuit-tracer` only if Week 3 finishes early. |

### Strategy-vs-code gaps to close first

- `backend/model.py`, `backend/main.py`, `backend/research.py` are GPT-2-specific: `model_name` default `gpt2-small`, layer bounds `le=11`, tied embedding assumption. All need parameterization for Llama.
- No ablation infrastructure exists anywhere. The `generate_steered` hook in `backend/research.py` adds activations but doesn't project them out. The ablation primitive (`h - (h·d̂)d̂` for a unit direction `d̂`) is the first new thing to build.
- No contrastive-pair dataset beyond the 8 sentiment pairs in `research.py::get_contrastive_pairs`. JailbreakBench + Alpaca subset per the Jan 2026 strategy doc is still the right starting dataset.

### What to deprioritize

- 3D PCA trajectories — flashy but hard to tie to causal claims. Defer.
- Polished visualization of individual attention heads — useful, but not the differentiator.
- Ambitious "multi-model supported" refactor — pick Llama-3.2-3B, go deep.
- Publication framing — this is a portfolio piece (per your own proposal). Don't trap yourself in a paper-scope that blocks shipping.

---

## 5. Open Decisions

These need answers before Week 1 starts. Calling them out explicitly because Jan 2026 strategy doc answered Q1–Q8 but the landscape has shifted enough that 2–3 of those answers deserve a second look.

1. **Target model.** `RESEARCH_STRATEGY.md` says Llama-3.2-3B. Tooling-side, Llama-3.2-1B is easier to ship browser-native (smaller, still refuses). If the tool side is leading, 1B may be the right call. If SAE work enters scope, Gemma-2-2B.
2. **Is Moon actually participating in the research direction?** The Jan 2026 docs assume a researcher co-lead. If you're driving alone, scope down — the 4-week plan above assumes one person.
3. **Are you applying to MATS/Fellows April 26?** If yes, the plan should compress: Week 1 must be enough of a demo for the application. If no, you have more runway.
4. **Attribution graphs — in scope or out?** It's the hottest technique but adds real engineering complexity. Default: out for v1, stretch goal for v2.
5. **Blog post target.** LessWrong / Alignment Forum is the default. If Moon is co-author, also submit to ICLR 2026 Blogpost Track or the ICML 2026 Mech Interp Workshop.

---

## 6. Key References (bookmark these)

### Methodology

- [Arditi et al., NeurIPS 2024 — Refusal Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) — baseline
- [Wollschlager et al., ICML 2025 — Concept Cones](https://arxiv.org/abs/2502.17420) — multi-direction extension
- [Zhao et al. (preprint; venue claim NeurIPS 2025 unconfirmed) — Harmfulness vs Refusal](https://arxiv.org/abs/2507.11878)
- [Luu et al., ACL 2025 **Findings** — COSMIC](https://arxiv.org/abs/2506.00085)
- [Prakash et al., Sept 2025 — Dissecting LLM Refusal](https://arxiv.org/abs/2509.09708)
- [LessWrong, March 2026 — Single Direction vs Low-Rank Refusal in Small LLMs](https://www.lesswrong.com/posts/LMkvjDTLKFrgdzJdG/single-direction-vs-low-rank-refusal-in-small-llms-1) — Llama-3.2-3B: ~15% compliance with one vector, ~37% with five-layer stack
- [Cheng et al., Apr 2026 — What Drives Representation Steering?](https://arxiv.org/abs/2604.08524) — direct on Llama-3.2-3B; vectors compress to 1–10% of dims
- [Maskey et al., rev Apr 2026 — Over-Refusal and Representation Subspaces](https://arxiv.org/abs/2603.27518) — over-refusal is higher-dim than harmful-refusal
- [Joad et al., Feb 2026 — There Is More to Refusal than a Single Direction](https://arxiv.org/abs/2602.02132)
- [Geiger et al., JMLR 2025 — Causal Abstraction](https://jmlr.org/papers/v26/23-0058.html)
- [Chen et al., Sept 2025 — Persona Vectors](https://arxiv.org/abs/2507.21509)
- [Ferrando et al., ICLR 2025 — Do I Know This Entity?](https://arxiv.org/abs/2411.14257)

### Circuits & attribution graphs

- [Anthropic — Circuit Tracing Methods (March 2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [Anthropic — On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [Open-source circuit tracer](https://www.anthropic.com/research/open-source-circuit-tracing)

### SAEs (read the skeptical take first)

- [Kantamneni et al., Feb 2025 — Are Sparse Autoencoders Useful?](https://arxiv.org/abs/2502.16681)
- [Use SAEs to Discover Unknown Concepts](https://arxiv.org/html/2506.23845v1)
- [Transcoders Beat SAEs for Interpretability](https://arxiv.org/html/2501.18823v1)
- [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)

### Tooling

- [Neuronpedia](https://www.neuronpedia.org/)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [SAELens](https://github.com/decoderesearch/SAELens)
- [circuit-tracer](https://github.com/decoderesearch/circuit-tracer)
- [Transformer Explainer (pedagogy reference)](https://poloclub.github.io/transformer-explainer/)

### Hiring / community

- [Anthropic Fellows Program 2026 (deadline April 26)](https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/)
- [MATS Summer 2026 (deadline April 26)](https://www.matsprogram.org/program/summer-2026)
- [Neel Nanda — Pragmatic Vision for Interpretability](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability)
- [ICML 2026 Mech Interp Workshop](https://mechinterpworkshop.com/)

---

*Compiled from three parallel research streams (landscape / tooling / adjacent directions) on April 17, 2026. Supersedes tactical recommendations in `RESEARCH_STRATEGY.md` where they conflict; preserves the overall research framing.*

*Amended April 28, 2026: see "Update — April 28 2026" near the top. Adds Cheng 2604.08524, Maskey 2603.27518, Joad 2602.02132. Reorders predictions (harmfulness probe leads). Switches primary target to Llama-3.2-1B. Reflects that the deploy infrastructure and ablation primitive are now built.*
