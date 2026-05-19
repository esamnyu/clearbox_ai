# Building a browser-native mech-interp toolkit: one engineer's read on the 2026 refusal-direction landscape

> *Draft — April 28 2026. ~1,000 words (sprint plan called for ~800; trim during your final pass). Three TBDs flagged inline; the experiment-result paragraph and the live URLs are the only things blocking publication.*

---

I'm a full-stack engineer at MiQ, teaching myself mechanistic interpretability after hours. Over the last few months I've been building **NeuroScope-Web**, a browser-and-Python tool that lets you load a small transformer, watch its internals, and intervene on them. This post covers three things: what's in the tool, what I've actually learned about the 2026 refusal-direction literature in the process of building it, and what I plan to do with both next.

## What I built

NeuroScope is hybrid by design. The frontend is React + `transformers.js` — load GPT-2 small locally in the browser, no server, no install. The backend is FastAPI + TransformerLens, used when an analysis genuinely needs real tensors (gradients, fine-grained hooks, multi-token caches). The split is the cheapest path to "one URL, one reproducible investigation," which is the engineering claim I'm trying to make.

The integrated views are:

- **Logit lens** — what the model "would predict" at every layer.
- **Attention heatmap + 12 × 12 head grid** — every head pre-classified by pattern (induction, previous-token, position).
- **Gradient-based token importance** — which input tokens most influence a target token's probability.
- **Steering vectors** — pick contrastive prompts, compute *v = mean(positive) − mean(negative)*, watch an α slider from −3 to +3 push generation along that direction.
- **Direction ablation** *(just shipped)* — project a direction out of the residual stream at a chosen layer: *h′ = h − (h · d̂) d̂*. This is the causal counterpart to steering. Instead of *adding* a vector, you *remove* the component along it. It's the primitive Arditi et al. used to argue refusal is *caused* by a single direction, as opposed to merely correlated with one.

Live demo: **{{TBD — paste HF Spaces / Vercel URL after deploy}}**
Code: **{{TBD — paste GitHub URL}}**

## What I read

The story I started with was Arditi et al. (NeurIPS 2024). Refusal in chat models is approximately a single direction in activation space; difference-of-means over a (harmful, harmless) prompt set extracts it; ablating it flips refusal. The methodology is clean and the result has replicated widely.

The 2025–26 picture is richer.

[Wollschlager et al. (ICML 2025)](https://arxiv.org/abs/2502.17420) showed refusal lives in a *polyhedral concept cone* of mechanistically independent directions, not a single line. The cone has structure: the directions inside it are not redundant duplicates of each other but causally distinct under intervention.

[Zhao et al. (preprint 2507.11878, venue claim under verification)](https://arxiv.org/abs/2507.11878) ran the natural follow-up experiment. After ablating Arditi's refusal direction on Llama-3-8B and Qwen-7B, the model stops *saying* "I can't help with that" — but a probe trained to detect *harmfulness* on the same residual stream still fires. Refusal and harmfulness are distinct directions. The standard ablation is a behavioral patch, not a causal-level intervention. It changes what the model says about its judgment, not the judgment itself.

In 2026 the picture sharpened around model size. A [March LessWrong post](https://www.lesswrong.com/posts/LMkvjDTLKFrgdzJdG/single-direction-vs-low-rank-refusal-in-small-llms-1) reported that Llama-family models *specifically* need low-rank bases: Llama-3.2-3B sits at ~15% compliance under a single ablated vector, ~37% under a five-layer stack. [Cheng et al. (April 2026, 2604.08524)](https://arxiv.org/abs/2604.08524) confirmed this on Llama-3.2-3B and added that the effective steering vectors compress to 1–10% of model dimensions while preserving the behavioral effect. [Maskey et al. (revised April 2026, 2603.27518)](https://arxiv.org/abs/2603.27518) reframed the rank > 1 phenomenon: a lot of it is about *over-refusal* (high-dim) versus *harmful refusal* (approximately single-direction), which is a different story than "safety tuning has multi-directional depth." [Joad et al. (Feb 2026, 2602.02132)](https://arxiv.org/abs/2602.02132) found per-category geometrically distinct directions that nevertheless collapse to a single operational knob.

The thing I take away — written in my own voice, not paraphrased from any one paper — is that "is refusal multi-directional?" is no longer a live question. It is. The live question is what the *structure of the failure* tells us. After you ablate the refusal cone, the harmfulness signal *is still there*, on the models Zhao tested. Whether the same holds on smaller Llamas is open.

## What's next — the experiment I'd actually like to run

The honest version of NeuroScope's first experiment is small. I haven't migrated the backend to Llama yet — it still runs on GPT-2 small, which fits the browser-loadability story but was never RLHF'd, so refusal isn't on the table. What *is* on the table is the sentiment direction baked into the existing contrastive pairs. As a proof of plumbing for the new ablation primitive, I extracted *v = mean(positive) − mean(negative)* on layer 6 and projected it out at the same layer. Result: **{{TBD — paste a sample baseline-vs-ablated generation pair plus the next-token Δlogprobs}}**. This is not a finding. It's evidence that the tool can run a causal intervention end-to-end.

The real experiment I want to run is the one Zhao 2025 left open on small models. Take **Llama-3.2-1B-Instruct** — the smallest open Llama that still actually refuses. Extract the refusal cone (Wollschlager's iterative method, ~50 lines on top of what NeuroScope already exposes). Ablate it. Then train a harmfulness probe on the residual stream — Zhao's setup — and run it on the held-out harmful prompts.

If the harmfulness signal survives ablation, that's the result: "refusal ablation" on the smallest open Llama is, like its 8B siblings, a verbal patch — confirming Zhao at the model size where independent researchers can actually iterate. If the signal disappears, that's *also* the result: you've found the model size at which safety tuning is genuinely integrated rather than glued on. Either outcome lands.

What's blocking me is a Llama backend port (the layer-bound config is currently `le=11`, GPT-2-specific; needs `le=15` for 1B), the iterative direction extractor, and a refusal-prompt dataset (JailbreakBench plus a benign Alpaca subset). The ablation primitive — the piece I'd have to write from scratch in any of these papers — already exists in the tool. That's the one durable advantage of doing this as a tool first.

If you've worked on small-Llama refusal-direction interp, or if you have opinions about whether Wollschlager's iterative ablation is the right method for sub-3B models, I'd love to compare notes — open an issue on the repo or reply on the LessWrong cross-post.

---

*NeuroScope-Web is open-source ({{repo URL}}). The {{strategy doc}} explains how the project's research framing has evolved from January 2026 to today; the {{landscape doc}} is the live brief on what's settled and what isn't in mech-interp's refusal line of work as of late April 2026.*
