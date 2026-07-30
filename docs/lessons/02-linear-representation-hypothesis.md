# Lesson 2 — Why Directions Encode Concepts

## The shift you need to make

The first instinct people have about neural networks is "neuron = concept." There's a grandmother neuron. There's a curve-detector neuron. Mech interp in 2026 mostly does not work that way. The unit of meaning is not a single coordinate of $h$. It is a **direction in vector space** — a unit vector $\hat{d}$ such that $h \cdot \hat{d}$ tells you how much of some concept is present.

This is the **linear representation hypothesis**. It is not a theorem. It is an empirical regularity that holds well enough, often enough, that nearly every technique in this codebase relies on it. If you are interviewing for an interpretability role, you must be able to state it, defend it, and acknowledge where it breaks.

## The classical evidence: word embeddings

The cleanest early demonstration came from word2vec ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)). Take the embedding vectors for `king`, `man`, `queen`, `woman`. Then:

$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

The vector $\text{king} - \text{man}$ encodes "royalty." Adding it to `woman` produces `queen`. This is **linear arithmetic on concepts.** The concept "royalty" exists as a direction, and that direction is preserved across many word pairs (queen-woman, king-man, prince-boy, princess-girl).

This was a property of word embeddings, not transformers. But the same phenomenon was found inside transformer residual streams. The paper to cite if asked: [Park, Choe, Veitch — *The Linear Representation Hypothesis and the Geometry of Large Language Models* (2023)](https://arxiv.org/abs/2311.03658). They formalize when "concept = direction" is well-defined (the unembedding-causal-inner-product framing) and show the property holds across many high-level concepts in Llama and GPT-2.

## Why this matters for every primitive in this codebase

Every technique you will study from Lesson 3 onward depends on this hypothesis:

- **Steering vectors** (Lesson 3) — adding a vector along a concept direction increases the concept's presence. This only works if the concept *is* a direction.
- **Projection-removal ablation** (Lesson 4) — subtracting $(h \cdot \hat{d}) \hat{d}$ removes the concept. Same assumption.
- **Linear probes** (Lesson 6) — fitting a logistic regression to predict "is this prompt harmful?" from the residual stream finds a hyperplane normal vector $w$ such that $w \cdot h$ is the harmfulness score. Same assumption.

When you read `extract_steering_vector` in `backend/research.py` and see this line:

```python
steering_vector = mean_pos - mean_neg
```

you are looking at the linear representation hypothesis in code. Subtract the mean residual of negative examples from the mean residual of positive examples; the result is the direction that points from "negative" to "positive" in the residual stream. The codebase computes this for sentiment, but the recipe is identical for refusal (Lesson 5) and the implementation in `backend/refusal_pairs.py` is intentionally a drop-in replacement.

## Superposition: the reason this works at all

`d_model = 768` for GPT-2. The model represents *vastly* more than 768 distinct concepts. How? **Superposition.** Concepts live in a lower-dimensional subspace each, and the directions are not orthogonal — they're packed in at small but nonzero angles. The model tolerates the interference because most concepts are not active at once.

The canonical paper: [Elhage et al., *Toy Models of Superposition* (Anthropic, 2022)](https://transformer-circuits.pub/2022/toy_model/index.html). They train tiny networks where ground truth is known, and watch the network pack 5 concepts into a 2D activation space at pentagonal angles. The geometry generalizes.

The practical consequence for us: when we extract a steering vector via difference-of-means, we are getting the **best linear estimator** of the concept direction, but it is not pure. It will contain some leakage from other concepts that co-occurred in the prompts. This is why contrastive datasets need to be carefully matched (Lesson 3) — to cancel out the leakage in the difference.

## When linearity fails

The hypothesis is not a free lunch. There are well-documented regimes where it breaks:

1. **Late layers.** Just before the unembedding, the model is doing token-prediction-specific arithmetic. Concept directions blur into token directions. This is why `extract_steering_vector` recommends middle layers (4–8 for GPT-2). The same caveat applies to Llama; Arditi's refusal direction is typically extracted from a mid-to-late layer chosen empirically (Lesson 3).
2. **Compositional concepts.** Negation, conditionals, and quantifiers are not always linear. "I want to bake a cake" and "I don't want to bake a cake" do not differ by a simple "want / don't want" vector; the negation interacts with the rest of the sentence.
3. **Multi-modal concepts.** [Wollschlager et al. (2025, arxiv 2502.17420)](https://arxiv.org/abs/2502.17420) — which you'll meet in Lesson 5 — find that refusal in Llama-family models is mediated by a **polyhedral cone of directions**, not a single direction. So even when concepts are linear, they may be **multi-direction** linear.

You should never claim "concepts are directions" in an interview as if it were a theorem. Say: *the linear representation hypothesis holds for many high-level features at intermediate layers; the codebase relies on it; recent work has refined it from "one direction" to "low-rank subspace."*

## A worked picture

Imagine the residual stream at layer 6 of GPT-2, restricted to 8 contrastive sentiment pairs from `get_contrastive_pairs()` in `backend/research.py`:

- 8 positive prompts → 8 vectors in $\mathbb{R}^{768}$
- 8 negative prompts → 8 vectors in $\mathbb{R}^{768}$

If sentiment is linear at this layer, the positive cluster and negative cluster are separated by a single hyperplane. The normal vector of that hyperplane — pointing from negative-mean to positive-mean — is the "sentiment direction." Project any new prompt onto this direction: positive number = positive prompt, negative number = negative prompt.

This is exactly what `extract_steering_vector` returns. Lesson 3 walks the code line by line.

## Check yourself

1. State the linear representation hypothesis in one sentence. Now state one experiment that would falsify it.
2. Why does superposition let a 768-dim residual stream represent thousands of concepts?
3. If sentiment lives along a direction $\hat{d}$, what does $h - 2(h \cdot \hat{d}) \hat{d}$ do to a positive-sentiment prompt? (Hint: it's not "remove" — it's a reflection.)

## Read next

Lesson 3 — `03-direction-extraction.md`. We walk through `extract_steering_vector` in `backend/research.py` and show how difference-of-means turns a contrastive dataset into a concept direction.
