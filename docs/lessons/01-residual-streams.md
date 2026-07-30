# Lesson 1 — Residual Streams: The Substrate of Interpretability

## The picture to hold in your head

A transformer is not a black box that maps tokens to tokens. It is a stack of **vector updates**. At every layer, every token position carries a vector of dimension `d_model`. Each transformer block reads that vector, computes a small update (an attention output, then an MLP output), and **adds** the update back. Output of block `L` for token `i` is:

$$h^{(L)}_i = h^{(L-1)}_i + \text{attn}^{(L)}(h^{(L-1)})_i + \text{mlp}^{(L)}(h^{(L-1)} + \text{attn}^{(L)}(h^{(L-1)}))_i$$

The vector $h_i$ — the running sum of all updates so far for token $i$ — is the **residual stream**. It is the through-line of the model: every block reads it, every block writes to it. The final residual stream at the last layer is what gets multiplied by the unembedding matrix to produce logits.

This framing comes from Anthropic's [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Elhage et al., 2021). It is the foundational mental model for mechanistic interpretability. Without it, the rest of the field looks like alchemy.

## Dimensions are not arbitrary

`d_model = 768` for GPT-2 small. `d_model = 2048` for Llama-3.2-1B. `d_model = 3072` for Llama-3.2-3B. These numbers are picked to balance capacity against compute, but the consequence for us is concrete: every token at every layer is a point in a $d_{model}$-dimensional vector space. **All of interpretability is geometry in that space.** Concepts are directions. Computations are projections, additions, and subtractions. The "polysemantic neurons" people complained about in 2022 were a symptom of trying to read individual coordinates instead of looking at directions.

When you see `d_model` mentioned in the codebase, picture the dimension of the room your residual stream lives in. The room is wide because the model needs to store many superposed concepts at once — see [Elhage et al., Toy Models of Superposition (2022)](https://transformer-circuits.pub/2022/toy_model/index.html) for the canonical treatment.

## How TransformerLens lets you read the stream

The whole reason this codebase uses [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) instead of raw HuggingFace is that TransformerLens names every intermediate tensor and lets you fetch any of them with a string. The naming convention you will see repeatedly:

- `hook_embed` — the token embedding before any block runs
- `hook_pos_embed` — the positional embedding
- `blocks.X.hook_resid_pre` — residual stream **entering** block `X`
- `blocks.X.hook_resid_mid` — residual stream **after attention, before MLP**
- `blocks.X.hook_resid_post` — residual stream **leaving** block `X` (after attention and MLP)

Open `backend/research.py` and search for `hook_resid_post`. You will find it in three places: `logit_lens` (line 108), `extract_steering_vector` (line 299), and `ablate_along_direction` / the steering hook (lines 425, 495). Every primitive in this project reads or writes the post-block residual stream. That is the surface you operate on.

The pattern is always the same:

```python
tokens, logits, cache = run_with_cache(prompt)
resid = cache[f"blocks.{layer}.hook_resid_post"]   # [batch, seq_len, d_model]
```

`cache` is a dictionary keyed by hook names. The value is a tensor whose shape is `[batch, seq_len, d_model]`. For a single prompt (`batch=1`), `resid[0, -1, :]` is the residual stream at the **last token** of layer `layer`. This particular index appears so often in the codebase that you should commit it to memory. We will explain *why* the last token in Lesson 3.

## Why this is the right substrate

Three properties make the residual stream the right place to do interpretability:

1. **Linearity.** Every block contribution is *added*. Nothing is overwritten. So if you know the residual at layer 5 and at layer 12, the difference is exactly the sum of contributions from blocks 5–11. You can decompose, attribute, and subtract.

2. **Bottleneck.** The residual stream is the **only** channel through which information flows between blocks. Whatever the model knows after block 5 is in $h^{(5)}$ — there is no hidden side-channel.

3. **Read-out is linear.** The final residual gets multiplied by the unembedding matrix $W_U$ to produce logits. That is, the model's prediction is a *linear function* of the final residual. This is why a "logit lens" (Lesson 1 sequel: applying $W_U$ to every layer's residual, not just the last) gives an interpretable signal.

The first two properties hold for any transformer with residual connections. The third is what makes mechanistic interpretability tractable at all. Without it we would be projecting through a final nonlinear classifier, and most of the techniques in this codebase — logit lens, steering, ablation — would lose their crisp meaning.

## A worked example: the Eiffel Tower

Type "The Eiffel Tower is in the city of" into the logit lens. Look at the last-token residual at each layer. Apply $W_U$, take softmax, look at the top-1 prediction:

- Layers 0–10: the model predicts boring fillers (`the`, `a`, `New`).
- Layer 11 (the final block in GPT-2 small): suddenly `Paris` jumps to top-1.

This is the residual stream "deciding" at the last possible moment. Nostalgebraist's original [logit lens post (2020)](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) showed this pattern across many prompts. Moon's notebook (which this codebase descends from) showed it for the Eiffel Tower specifically. It is a useful prompt because the answer is one token and lives in a clearly factual subspace.

The lesson is not "models compute at the last layer." It is that **the residual stream is where you watch the computation happen.** Every later primitive in this curriculum is just a way to extract something specific from this stream.

## Check yourself

1. Why is the residual stream additive across blocks, and what would change if it were multiplicative?
2. If `d_model = 2048` for Llama-3.2-1B, how many "directions" can the model linearly separate? (Hint: think Johnson-Lindenstrauss; the answer is exponentially many, not 2048.)
3. What is the shape of `cache["blocks.5.hook_resid_post"]` for a 1-prompt batch of 7 tokens on GPT-2 small?

## Read next

Lesson 2 — `02-linear-representation-hypothesis.md`. Now that you know *where* concepts live (in the residual stream), we ask *how* they're encoded: as directions, not as neurons.
