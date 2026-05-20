# Lesson 3 — Difference-of-Means: Extracting a Concept Direction

## The recipe in one line

Given a set of prompts that exhibit a concept and a set that does not, the **concept direction** is the difference of mean residual streams:

$$v = \frac{1}{|P|} \sum_{p \in P} h^{(L)}_{\text{last}}(p) \;-\; \frac{1}{|N|} \sum_{n \in N} h^{(L)}_{\text{last}}(n)$$

That is the whole technique. The remaining 1000 words explain why each piece is the way it is.

## The code, walked

Open `backend/research.py` and find `extract_steering_vector` (line 271). Strip it to the essential five lines:

```python
pos_activations = [get_last_token_activation(p, layer) for p in positive_prompts]
neg_activations = [get_last_token_activation(n, layer) for n in negative_prompts]
mean_pos = torch.stack(pos_activations).mean(dim=0)
mean_neg = torch.stack(neg_activations).mean(dim=0)
steering_vector = mean_pos - mean_neg
```

And the helper:

```python
def get_last_token_activation(prompt, layer_idx):
    _, _, cache = run_with_cache(prompt)
    resid = cache[f"blocks.{layer_idx}.hook_resid_post"]
    return resid[0, -1, :]   # [d_model]
```

That's it. Five lines of math, three lines of plumbing.

## Why the last token

`resid[0, -1, :]` — batch 0, last token position, all `d_model` coordinates. Why the **last** token?

Because GPT-2 (and Llama, and every causal LM in this codebase) is autoregressive. Each token can only attend to itself and earlier tokens. The last token is therefore the only position that has "seen" the entire prompt. By layer `L`, the last-token residual has aggregated context from every preceding token.

If you extract from token position 0 you only see information about the first token. If you extract from token 3 you see the first four tokens. If you extract from `-1` you see everything. For prompt-level concepts — sentiment, refusal-versus-compliance, harmfulness — you want everything.

There is one subtlety. In Llama-3.2-Instruct, the prompt is wrapped in chat-template special tokens (`<|begin_of_text|>`, `<|start_header_id|>`, etc.) — see `apply_chat_template` at line 28 of `research.py`. The "last token" is the **assistant turn start token**, the position from which the model would begin generating its reply. This is exactly the right position to probe for refusal: it is where the model decides whether to comply or refuse. The codebase calls `apply_chat_template` from generation paths (`generate_steered`, `ablate_along_direction`) and from extraction on instruct models. It deliberately does *not* call it from `logit_lens` or `extract_steering_vector` for non-instruct models, because GPT-2 has no chat template.

## Why difference-of-means, and not something fancier

You might wonder: why not fit a logistic regression and use the learned weights as the direction? Or train a small classifier? Difference-of-means looks almost too simple.

Two reasons it survives:

1. **It is the unbiased estimator of the concept direction under the linear representation hypothesis** (Lesson 2). If $h = c \cdot \hat{d} + \text{noise}$ where $c$ is a scalar "amount of concept," and the noise is mean-zero and uncorrelated with $c$, then $\text{mean}(h_{\text{positive}}) - \text{mean}(h_{\text{negative}})$ is proportional to $\hat{d}$ in expectation. Logistic regression gives you a related vector, but it bakes in classification thresholds you don't actually want.

2. **It is robust.** With only 8 contrastive pairs you cannot reliably fit a high-dimensional classifier — overfitting will dominate. Means converge faster than discriminative weights. This is empirically why the refusal-direction literature (Arditi 2024 onward) standardized on difference-of-means.

Cite: [Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (NeurIPS 2024, arxiv 2406.11717)](https://arxiv.org/abs/2406.11717). Their method on Llama-3-8B uses difference-of-means on JailbreakBench + Alpaca prompts at the last-input-token position of an instruct-model template. The codebase's `extract_with_ablation` in `backend/refusal_bench/harmfulness_probe.py` is a direct port of this pipeline to TransformerLens.

## Why length-matching matters

Look at `get_contrastive_pairs` (line 249 of `research.py`):

```python
("I think this movie is amazing", "I think this movie is terrible"),
("The food at this restaurant is delicious", "The food at this restaurant is disgusting"),
...
```

Each pair has the same number of tokens. Why? Because the residual at position `-1` depends on **position**. If your positive prompt is 6 tokens and your negative prompt is 7, you're comparing the last-token residual at position 5 to the last-token residual at position 6. Those two positions have different positional embeddings, and they aggregate different amounts of context. The difference will pick up a position artifact along with the sentiment.

When you length-match — same token count, same syntactic structure, differing only in the concept word — the position effect cancels in the difference. This is the same reason `backend/refusal_pairs.py`'s docstring requires "per-pair token-count delta within 20% on the Llama-3.2-1B-Instruct tokenizer." Twenty percent is generous because the refusal/harmless contrast tolerates more length variation than tight sentiment pairs, but it is not zero.

The cleaner the pair construction, the cleaner the extracted direction. This is the dirty secret of interpretability: most of the work is in the dataset, not the algorithm.

## Layer choice

`extract_steering_vector` takes a `layer` argument. Why?

The same concept can be encoded at different depths. Early layers (0–3 in GPT-2) carry mostly token-level and syntactic features. Late layers (10–11) carry features that are entangled with next-token prediction. **Middle layers** tend to carry the cleanest high-level concept representations.

Moon's notebook (the codebase ancestor) defaulted to layer 6 of GPT-2. For Llama-3.2-1B (16 layers) the equivalent middle is roughly layer 8–10. Arditi 2024 sweeps all layers and picks the one with the strongest behavioral effect; you should do the same when extending the codebase to Llama.

The `layer` argument is mirrored in `extract_with_ablation` (line 142 of `harmfulness_probe.py`), which deliberately separates `layer_extract` (where to read the residual) from `ablation_layer` (where to intervene). This is so you can probe **downstream** of the intervention site to see how a signal propagates — useful in Lesson 7 when we ask whether a harmfulness signal survives a refusal ablation at the same layer it was originally extracted from.

## What the returned vector means, geometrically

`steering_vector = mean_pos - mean_neg` is a vector in $\mathbb{R}^{d_{model}}$. Its **norm** tells you how strongly the concept is encoded — the codebase reports `vector_norm` in the JSON response (line 317) for exactly this reason. A small norm means the concept barely registers in this layer; a large norm means it's vivid. Its **direction** is the concept axis. The unit vector $\hat{d} = v / \|v\|$ is what gets used downstream:

- For **steering** (Lesson 4 detour into `generate_steered`), you add `alpha * v` (the un-normalized vector) to the residual stream during generation.
- For **ablation** (Lesson 4, main path), you project the residual onto $\hat{d}$ and subtract.

Steering and ablation are duals: addition along a direction versus projection-removal of that direction. The same extracted direction powers both.

## Check yourself

1. Why does extracting from token position 0 give you a useless direction for sentiment, but a meaningful direction for "this prompt starts with a vowel"?
2. If you length-mismatch your contrastive pairs by 5 tokens, what does the difference vector pick up that you didn't ask for?
3. You extract a sentiment direction at layer 6 with norm 2.3. The same recipe at layer 11 gives norm 8.7. Does layer 11 represent sentiment more strongly? (Hint: think about what late-layer norms confound.)

## Read next

Lesson 4 — `04-projection-removal.md`. The ablation primitive: how subtracting a single rank-1 projection from the residual stream changes the model's behavior — and why that change is causal, not correlational.
