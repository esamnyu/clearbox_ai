# 9-Day Sprint: Anthropic Fellows Application

> **Target:** Submit May 2026 cohort application by **Sat April 26, 2026**.
> **Starting:** Thu April 17, 2026 (today).
> **Runway:** 9 calendar days, ~30–35 focused hours around a day job.
> **Philosophy:** do NOT try to migrate to Llama. Deploy what exists, ship one small real result, write well, submit.

---

## The bet

| In scope | Out of scope |
|---|---|
| Deploy current GPT-2 tool to a live URL | Llama-3.2 migration |
| Add **one** ablation primitive (projection removal) on top of existing steering code | Full Arditi replication |
| Run **one** small experiment on the sentiment contrastive pairs | Attribution graphs / circuit tracer integration |
| Write **one** short blog post (~800 words) | Refactoring the codebase |
| Polish README + record 3-min Loom | Docs overhaul |
| Draft + polish research statement | Additional research papers |

**Scope discipline note:** every hour you spend on out-of-scope items is an hour not spent on the statement. The statement is the highest-leverage artifact by far.

---

## Calendar

### Thu Apr 17 (today) — evening, 2h
- [ ] Create HuggingFace account if none; reserve a Space name: `ethan-sam/neuroscope-api`
- [ ] Create Vercel account; link GitHub
- [ ] Skim `docs/DEPLOYMENT.md` end-to-end. Identify any credentials you don't yet have.
- [ ] Write out your 1-sentence research question (see `FELLOWS_RESEARCH_STATEMENT.md` §1). Commit it to a scratchpad — you will rewrite it 4 times this week.

### Fri Apr 18 — weeknight, 2–3h
- [ ] **Deploy backend to HuggingFace Spaces.** Follow `docs/DEPLOYMENT.md` §1. CPU tier is fine for GPT-2 small.
- [ ] Verify `/load`, `/logit-lens`, `/attention` endpoints respond from the public URL.
- [ ] *If blocked* — note the blocker, timebox to 1h more, then switch to Vercel frontend deploy and come back to backend tomorrow.

### Sat Apr 19 — full day, 6–8h
**Morning (3–4h) — Deploy + connect**
- [ ] Deploy frontend to Vercel. Point it at the HF Spaces backend URL via env var.
- [ ] Full end-to-end smoke test: load from public URL, enter prompt, see activations.
- [ ] Fix CORS / env issues.

**Afternoon (3–4h) — Ship one experiment**
- [ ] Add **one endpoint** to `backend/research.py`: `ablate_along_direction(prompt, direction, layer)`.
  - Implementation is ~20 lines: project the residual stream onto the direction and subtract. Hook it at the target layer.
  - Formula: `h' = h - (h · d̂) d̂` where `d̂` is the unit-normalized direction.
- [ ] Run it on the existing sentiment steering vector. See what happens to generation when you ablate the positive-vs-negative sentiment direction.
- [ ] Record: one before/after generation pair, the numerical change in next-token probabilities.
- [ ] This is not a novel finding. It's evidence that your tool can do causal interventions.

### Sun Apr 20 — full day, 6–8h
**Morning (3–4h) — Blog post + Loom**
- [ ] Draft an 800-word post. Title candidate: *"Building a browser-native mech-interp toolkit: one engineer's read on the 2026 refusal-direction landscape."*
  - 200 words: what you built + link to live demo
  - 400 words: honest summary of the 2026 landscape (Arditi → Wollschlager → Zhao), written in your own voice using `RESEARCH_LANDSCAPE_2026.md` as a source
  - 200 words: the small experiment you ran and what's next
- [ ] Record 3-minute Loom walkthrough. Script: open the live URL → type a prompt → show logit lens → show attention → show ablation. Don't narrate what it does; narrate *why this slice of the tool is useful to a researcher*.

**Afternoon (2–3h) — Research statement v1**
- [ ] First full draft of research statement using `FELLOWS_RESEARCH_STATEMENT.md`.
- [ ] Don't edit yet. Just write through.

**Evening (1h) — README polish**
- [ ] Apply the 60-second-skim structure from `docs/DEPLOYMENT.md` §5.

### Mon Apr 21 — weeknight, 2–3h
- [ ] Read Sunday's research statement cold. Mark every sentence that says nothing.
- [ ] Rewrite §3 (Research Program). Make every claim specific.
- [ ] Get a friend to read it. Note: the friend should be someone who will say "this is vague" when it is vague.

### Tue Apr 22 — weeknight, 2–3h
- [ ] Incorporate friend's feedback.
- [ ] Publish blog post (Substack, personal site, or LessWrong). Link from README.
- [ ] Final README pass. Reviewers will skim this in 60 seconds.

### Wed Apr 23 — weeknight, 2–3h
- [ ] Second friend reads the statement. Prefer someone closer to ML research if you can.
- [ ] Final polish pass.
- [ ] Line up references if required. Name two people who can credibly speak to your engineering or research potential.

### Thu Apr 24 — weeknight, 2–3h
- [ ] Fill out the Fellows application form. Copy in statement. Attach links.
- [ ] Test every link in a clean incognito browser.
- [ ] Read the full application once. Leave it overnight. Do not submit yet.

### Fri Apr 25 — weeknight, 1–2h
- [ ] Morning coffee read. Fix any final awkwardness.
- [ ] **Submit.** 24h before deadline. Do not be a last-minute submitter for this one.

### Sat Apr 26 — buffer day
- [ ] Used only if Friday went wrong.

---

## Hour-budget reality check

Total: ~30 hours across 9 days.

- Deploy + connect: 5–8h
- One-experiment endpoint: 2–4h
- Blog post + Loom: 4–5h
- Research statement (draft + 3 revisions): 8–10h
- README polish + application form: 2–3h
- Unplanned buffer: 3–5h

If you find yourself with extra capacity, spend it on the statement, not on new features.

---

## Red-flag tripwires

- **Day 3 (Sat evening) and the backend still isn't deployed.** Stop migrating; switch to a simpler deploy target (Fly.io with a Dockerfile) or skip the ablation endpoint and deploy what works. The live URL matters more than the new feature.
- **Day 5 (Mon) and the research statement isn't drafted.** You are out of time for exploratory writing. Write whatever comes out and revise.
- **Day 7 (Wed) and friends haven't read it.** Ask publicly on Slack / Discord / LessWrong. Feedback from a stranger in the field is better than no feedback.
- **Day 8 (Thu) and you're still coding.** Stop. Submit.

---

## What you'll have submitted

- Research statement (~800 words, specific, honest, cites 2024–26 papers)
- Live demo URL, working on first click
- GitHub repo, skimmable README, blog post linked
- 3-min Loom walkthrough
- References lined up

That is a credible Fellows application. It is not a strong one by research-publication standards, but it is strong by *portfolio + taste + engineering* standards — which is the lane Anthropic explicitly hires for.

---

## After submission

- Apply July cohort too. May rejection doesn't block July. Between now and then, you can ship the failure-mode audit for real on Llama-3.2-3B using the plan in `docs/RESEARCH_LANDSCAPE_2026.md` §4.
- Send a cold email to one person at Anthropic or an alum of Fellows / MATS asking for a 15-minute chat. Reference your blog post and tool. People respond to specific artifacts more than to generic outreach.

---

*Last updated: April 17, 2026. Revise this file if the plan slips. A slipped plan honestly tracked is better than a tidy plan quietly ignored.*
