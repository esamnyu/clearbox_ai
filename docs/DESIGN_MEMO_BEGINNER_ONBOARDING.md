# Design memo — beginner-onboarding pass

> May 20, 2026. Companion to the editorial-redesign commits.

## The problem

NeuroScope-Web's editorial identity is an asset for the target audience (an Anthropic Interpretability reviewer who reads transformer-circuits.pub for breakfast). It is a *liability* for a first-time visitor who has never heard of mechanistic interpretability. The Roman-numeral sections, the Fraunces wordmark, and the seven-panel dashboard with terms like *"logit lens"*, *"head classification"*, *"steering vector"* expect domain fluency they don't have.

A first-time visitor lands on the URL, sees seven sections, and has no clue what any of it does or where to start. The application's headline finding — the Refusal Bench dissociation — lives at **VII**. Most visitors never scroll there.

## The rejected solution

A SaaS-style guided tour with chunky "Next →" modals, a chatbot mascot, or a "Welcome 👋 Let's get started" hero card would land the right information in the wrong register. The editorial voice rejects all three. Anything that reads as "onboarding flow" undermines the trust the design is trying to earn.

## The chosen solution

Borrow from editorial publications, which routinely onboard non-expert readers via **narrative pacing, decklines, sidebars, and footnotes**. Adapt that vocabulary verbatim.

Four primitives, all manuscript-native:

### 1. **Welcome deck** (`<WelcomeDeck />`)

A short italic deck below the masthead, above Section I. Two paragraphs of Newsreader italic at body-size, no chrome. Three sentences in the first paragraph telling the reader what this tool is for. Four numbered hints in the second paragraph — *i. type, ii. click, iii. scroll, iv. read* — that route a reader through one concrete experiment without ever using the word "tutorial." A drop-cap on the first letter is the single visual flourish.

Persistence: shown to every visitor. Dismissing the deck is not a primary interaction; a small *"hide intro"* link at the end of the second paragraph stores `neuroscope-onboarded=1` in `localStorage` and collapses the deck to a single recoverable line. Repeat visitors don't pay the deck's vertical cost; newcomers get oriented without a takeover.

### 2. **Defined term** (`<DefinedTerm />`)

The footnote pattern. Any technical term in any header is wrapped in this primitive. Visually:

- The term itself, with a **dotted underline 4px below the baseline** (matching manuscript convention for a footnoted phrase)
- A small **superscript dagger (`†`)** in slate-500 immediately after the term — visual signal that more is available

On **hover** or **keyboard focus** (and on **click** for touch), a small paper-tone popover unfolds beneath the term. The popover contains:

- The plain-English definition in italic Newsreader, ~14px, two sentences max
- A `→ Lesson N` link in mono, vermillion, that opens the relevant lesson in `docs/lessons/` in a new tab

Motion: 220ms fade-in, no slide. Popover sits *below* the term, not above, so it doesn't obscure the section header. Backdrop: `bg-paper` with `border-rule`. No drop shadow; we're not pretending these are physical objects.

A11y: keyboard-focusable button semantics; `aria-describedby` points at the popover; `aria-expanded` toggles on click for mobile.

### 3. **Try this** (`<TryThis />`)

A single line of italic Newsreader directly below the section header, before any controls. Format:

```
¶  Try: <concrete-suggestion>.  <expected-outcome>.
```

The pilcrow (`¶`) is the only ornament. The hint is in graphite; the outcome is in slate-500. Reads like a curator's note, not a tutorial instruction. One per section, except where the section is already self-evident.

### 4. **Refusal Bench deck** (added inline to `<RefusalBenchLeaderboard />`)

Two sentences of italic Newsreader under the section header, before the leaderboard renders. The deck:

> *Each row tests one published method for stopping the model from refusing harmful requests. The two columns measure (1) did the model stop saying "I refuse"; (2) did its internal sense that "this is harmful" actually disappear. When the two diverge, the method only suppressed speech, not understanding.*

The column headers themselves also get `<DefinedTerm />` wrap (Δ refusal rate, Δ AUC) for users who skip the deck.

## Copy voice

- Editor at a small literary-tech journal. *"For the curious."* *"A note for the reader."* *"Open with…"*
- No exclamation marks anywhere in the redesigned chrome
- No first-person plural for hand-holding (*"let's"*, *"we'll"*)
- Definitions never start with *"a [term] is…"* — start with what it does, not what it is. *"How each token looks at the others"* beats *"Attention is a mechanism for…"*
- Use the future tense sparingly. *"Type 'The Eiffel Tower is located in', then click generate."*

## Motion

The same restraint as the rest of the site: one orchestrated moment per interaction, then stillness.

- WelcomeDeck: drop-cap fades + scales in over 480ms on initial page load (single staggered moment)
- DefinedTerm popover: 220ms fade
- TryThis: no motion (static editorial detail)
- Section title hover (when DefinedTerm wrapped): underline lifts 2px on hover, 160ms

## Where it does NOT change anything

- The existing typography (Fraunces / Newsreader / JetBrains Mono) — unchanged
- The semantic color tokens (vermillion / cerulean / ink / paper / rule / graphite) — unchanged
- AblationHero (Section VI) — unchanged, still the visual centerpiece
- All panel internals — unchanged; the onboarding lives in the chrome around them
- Routing, state management, API surface — unchanged

## Files added / modified

- new — `src/components/WelcomeDeck.tsx`
- new — `src/components/DefinedTerm.tsx`
- new — `src/components/TryThis.tsx`
- patched — `src/components/RefusalBenchLeaderboard.tsx` (header deck + column-header DefinedTerms)
- patched — `src/App.tsx` (renders WelcomeDeck; `Section` accepts `tryThis` and ReactNode titles)
- patched — `src/index.css` (component classes for the popover)

## What this does NOT yet add

Two follow-ups deferred to a later pass:

1. **Panel-internal DefinedTerms** — wrapping terms like "steering vector," "harmfulness probe," and "residual stream" inside individual panel bodies. The primitive supports this; the wiring is just repetitive.
2. **Persistent reading state** — recording which lessons a user has opened, marking unread definitions. Out of scope for the first onboarding pass.
