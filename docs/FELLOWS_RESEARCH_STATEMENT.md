# Anthropic Fellows — Research Statement Outline

> Template for Ethan, May 2026 cohort. Deadline: April 26, 2026.
> Fill in the `{{BRACES}}` prompts. Delete all guidance text before submitting.
> Target length: ~1–1.5 pages (≈800 words). Anthropic explicitly values concision — do not pad.

---

## Before you start

**The frame:** you are not a research-trained applicant. Don't try to sound like one. Your edge is that you can ship, you read the field honestly, and you are pivoting into research with a clear question. Play that honestly.

**Common mistakes to avoid:**
- Hype about AI safety importance. Everyone writes this. Skip it.
- Generic "I've always been interested in interpretability." They don't care about feelings; they care about what you've done.
- Listing papers you've read. Cite them only when they shape a specific claim.
- Pitching a massive research program. Pitch one question + one method + why it's tractable.
- Hiding your engineering background. Your full-stack dev experience is an asset against the typical applicant pool, not a weakness.

**What reviewers actually look for** (from public signals — MATS reviewer guides, Fellows cohort-1 outcomes, 80K Hours analysis):
1. Can you identify a specific, tractable research question?
2. Can you ship work? (Public artifacts >> claims.)
3. Are you calibrated about the field? (Do you know what's recent, what's contested, what's settled?)
4. Will you collaborate well with a mentor for 3–6 months?

Your landscape brief (`docs/RESEARCH_LANDSCAPE_2026.md`) is evidence for #3 and #1. NeuroScope (deployed + cleaned up) is evidence for #2.

---

## Section 1 — Opening (≈100 words)

**Purpose:** Hook. What research question you're drawn to and why it's live in 2026.

**Template:**

> Over the past {{N months}}, I've been building NeuroScope — an in-browser mechanistic interpretability tool — while self-teaching the underlying research. The question I've ended up most interested in: **{{your 1-sentence research question}}**.
>
> Two recent results make this the right question to ask now. {{Cite Wollschlager 2025 and Zhao 2025 in 1 sentence each, connecting each to why the single-direction refusal story is incomplete.}} I want to spend the Fellowship testing {{specific prediction}} on {{specific model — Llama-3.2-3B or Gemma-2-2B}}.

**Recommended question for you:**
*"When the single-direction refusal claim breaks down — as it does on small Llama models — does the residual harmfulness signal after ablation reveal that current 'refusal ablation' techniques are behavioral patches, not causal interventions?"*

Concrete, tractable, current, directly addresses a gap between Arditi 2024 and the 2025–26 follow-ups (Wollschlager, Zhao, Cheng 2604.08524, Maskey 2603.27518). The headline contribution is the **harmfulness probe after ablation**, not the rank-k structure (which is now well-replicated).

---

## Section 2 — Background (≈150 words)

**Purpose:** What you've actually done, told in specifics. No CV-speak.

**Template:**

> I'm a full-stack engineer by trade ({{MiQ, role}}, {{N years}} of TypeScript/React/Python in production). NeuroScope started as a learning project: a browser-native toolkit pairing {{transformers.js inference / FastAPI + TransformerLens backend}} with React visualizations for GPT-2's internals.
>
> What I built: {{1–2 sentence technical summary — residual stream extraction, logit lens, attention heatmaps, steering vectors, 3D PCA trajectories, live deployed at {{URL}}}}. What I learned in the process of building it: {{1–2 sentences — the specific methodological thing that surprised you, e.g., "how much of 'interpretability' reduces to the quality of your contrastive dataset," or "the gap between activation-magnitude visualizations and actual causal claims"}}.
>
> Consultation with a Google DeepMind researcher in January pointed me toward the gap between what the tool could show (correlations, magnitudes) and what would be a real causal claim (ablation that flips behavior).

**Prompts to fill in:**
- One specific technical decision you made and why
- One specific thing you got wrong initially and corrected
- Why you're in this space despite not being a research grad

---

## Section 3 — Research Program (≈250 words)

**Purpose:** The actual proposal. This is what the reviewers grade most heavily.

**Template:**

> **The baseline.** Arditi et al. (NeurIPS 2024) showed refusal can be isolated as a single direction via difference-of-means over contrastive (harmful, harmless) pairs, and that ablating this direction causally flips refusal behavior. The methodology is clean and the result has replicated widely.
>
> **What's uncertain in 2026.** {{Summarize in 2–3 sentences: Wollschlager (ICML 2025) introduced concept cones — refusal lives in a polyhedral subspace, not a line. Zhao (preprint, 2507.11878) found that ablating the refusal direction on Llama-3-8B / Qwen-7B suppresses verbal refusal but leaves the model's internal harmfulness judgment detectable. The March 2026 LessWrong replication shows Llama-family models specifically need rank-k bases (Llama-3.2-3B: ~15% compliance with one vector, ~37% with a 5-layer stack). Cheng et al. (Apr 2026, 2604.08524) confirm and extend on Llama-3.2-3B; Maskey et al. (rev Apr 2026, 2603.27518) reframe rank > 1 as a property of *over-refusal* rather than safety-tuning depth — which sharpens what a compliance-vs-k curve actually means.}}
>
> **What I'd test during the Fellowship.** Three concrete, falsifiable predictions, ordered by novelty:
>
> 1. **(Headline)** On Llama-3.2-1B and 3.2-3B, ablating the refusal direction (or refusal cone) leaves a detectable harmfulness signal in the residual stream — replicating Zhao 2507.11878, which tested only 8B+ models. If yes, current "refusal ablation" techniques are behavioral patches, not causal interventions, and the result transfers to the model size most accessible to independent researchers.
> 2. **(Replication + extension)** On Llama-3.2-1B and 3.2-3B, the refusal subspace has rank k > 1; report a compliance-vs-k curve. Replicates the qualitative finding from Wollschlager + LessWrong + Cheng 2604.08524 on a smaller model where the data has not been collected.
> 3. **(Future work)** Subspace-rerouting jailbreak evasion (Winninger 2503.06269) of the multi-direction ablation. Defer if scope tightens.
>
> **Why this is tractable.** Extraction, visualization, and the projection-based ablation primitive (`h - (h·d̂)d̂`) are already built into NeuroScope. The net-new engineering is the migration to Llama-3.2-1B/3B and an iterative / orthogonalized ablation variant. The dataset is off-the-shelf (JailbreakBench + Alpaca subset). Compute is consumer-GPU scale; 1B is plausibly browser-loadable, which couples the research finding directly to the engineering thesis.
>
> **Why it's worth a Fellowship.** {{State honestly: a Fellowship would give you mentorship to turn tooling-with-a-question into publishable methodology. Attribution-graph overlay via the open-source circuit-tracer is a natural Anthropic collaboration. If you don't have a clear mentor in mind, say so and name the lab direction — "Interpretability team, particularly work coming out of the Circuit Tracing line" — rather than a specific person unless you've talked to them.}}

**Reviewer checklist for this section:**
- Specific model named? ✓
- Falsifiable prediction stated? ✓
- Method named, not waved-at? ✓
- Compute budget realistic? ✓
- One citation per claim, not five? ✓

---

## Section 4 — Fit (≈100 words)

**Purpose:** Why Anthropic specifically. Keep tight — this is filler if overdone.

**Template:**

> Three reasons Anthropic is the right place for this work:
>
> 1. **The Circuit Tracing line.** Attribution graphs are the natural next step for any refusal-direction work; {{Ameisen, Lindsey, et al.}} have built the tooling I would extend. Collaboration on the open-source tracer is a concrete asset I could contribute back to.
> 2. **Engineering culture.** The Interpretability team has been explicit that engineering is the bottleneck. {{Cite: Anthropic interpretability hiring page}}. My portfolio is stronger on engineering than on publications; a Fellowship is the right matching.
> 3. **{{Your specific reason — pick one and make it true. Don't invent.}}**

**Don't write:** "I love Anthropic's mission." Everyone does.
**Do write:** something you'd only know if you'd read their work.

---

## Section 5 — Closing (≈50 words)

**Purpose:** What happens after the Fellowship. Signals commitment + direction.

**Template:**

> After the Fellowship, I'd aim to {{one of: (a) continue in interpretability full-time — lab or independent, (b) apply the methodology work to evals / safety-relevant tasks at scale, (c) pursue a PhD in this area}}. NeuroScope will stay public regardless; the tool is already more useful as a teaching artifact than as a research claim, and I'd like to keep it that way.

---

## Artifacts to link (in application, not in statement prose)

- GitHub repo: `{{your-github}}/clearbox_ai` (clean README, tests passing, CI green)
- Live demo: `{{deployed URL}}`
- 3-minute Loom: tool walkthrough
- Blog post (short, honest): the one finding from the 9-day sprint

Reviewers skim in 60 seconds. Artifacts must work on first click. A broken demo is worse than no demo.

---

## Self-review checklist before submission

- [ ] Word count ≤ 1000
- [ ] No unsourced claims
- [ ] Every citation is from 2024 or later
- [ ] One specific, falsifiable research question named
- [ ] One specific model + dataset + method named
- [ ] No "I'm passionate about" language
- [ ] No bragging; no false modesty either
- [ ] Read aloud — does it sound like a person or a template?
- [ ] A friend in the field has read it
- [ ] Live demo works from a clean browser

---

*If you want me to review a draft once you've filled it in, paste it into a reply and I'll flag sections that need tightening.*
