import { useEffect, useState } from "react";

const STORAGE_KEY = "neuroscope-onboarded";

/**
 * Editorial deck below the masthead and above Section I.
 *
 * Sets context for first-time visitors in two paragraphs of italic
 * Newsreader: one paragraph framing the project, one paragraph routing
 * the reader through a concrete first experiment.
 *
 * A drop-cap on the opening word is the single visual flourish; it
 * fades + scales in over 480ms on first paint.
 *
 * Persistence: a discreet "hide intro" link sets localStorage and
 * collapses the deck to a single recoverable line. Repeat visitors
 * don't pay the vertical cost; first-timers get oriented.
 */
export default function WelcomeDeck() {
  // Start expanded; collapse to a single line if the visitor has
  // previously dismissed it.
  const [collapsed, setCollapsed] = useState<boolean | null>(null);

  useEffect(() => {
    // Hydration-safe read; default to expanded on SSR / first paint.
    const stored =
      typeof window !== "undefined" && window.localStorage.getItem(STORAGE_KEY);
    setCollapsed(stored === "1");
  }, []);

  const hide = () => {
    try {
      window.localStorage.setItem(STORAGE_KEY, "1");
    } catch {
      /* private mode etc.; collapse in-session only */
    }
    setCollapsed(true);
  };

  const show = () => {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* noop */
    }
    setCollapsed(false);
  };

  // null = pre-hydration; render the expanded form to avoid flicker for
  // first-time visitors.
  if (collapsed === true) {
    return (
      <aside className="mb-14 border-y border-rule py-4">
        <p className="font-serif text-sm italic text-slate-500">
          <span className="not-italic mr-2 text-slate-700">¶</span>A
          reader&apos;s intro is hidden.{" "}
          <button
            type="button"
            onClick={show}
            className="font-display not-italic underline decoration-dotted underline-offset-4 hover:text-vermillion-light"
          >
            show intro
          </button>
        </p>
      </aside>
    );
  }

  return (
    <aside className="mb-16 border-y border-rule py-8 sm:py-10">
      <p
        className="font-serif text-[1.05rem] italic leading-[1.7] text-slate-300 sm:text-[1.1rem]"
        style={{ animation: "reveal 600ms ease-out backwards" }}
      >
        <span
          aria-hidden
          className="float-left mr-3 mt-1 select-none font-display not-italic leading-[0.85] text-vermillion-light"
          style={{
            fontSize: "3.5rem",
            fontVariationSettings: '"opsz" 144, "SOFT" 30',
            animation: "reveal 720ms 80ms ease-out backwards",
          }}
        >
          A
        </span>
        language model is normally a black box: text goes in, text comes out.
        Mechanistic interpretability is the practice of opening it — reading the
        numbers passing between its layers to work out <em>why</em> it said what
        it said. This page is a workbench for doing that on GPT-2, live, in this
        tab. Nothing is uploaded; the model runs on your own machine.
      </p>

      <p className="mt-5 font-serif text-[1.05rem] italic leading-[1.7] text-slate-300 sm:text-[1.1rem]">
        It is also an experiment. Six published techniques claim to remove a
        safety-trained model&apos;s ability to refuse. §VII runs all six against
        the same test and asks a sharper question than &ldquo;did it stop
        refusing?&rdquo; — it asks whether the model still <em>knows</em> the
        request was harmful. The finding, stated plainly below, is that stopping
        the refusal and removing the knowledge are not the same thing, and the
        published methods mostly do the first.
      </p>

      <FindingCallout />

      <ol
        className="mt-7 space-y-3 font-serif text-[0.95rem] italic leading-relaxed text-slate-400 sm:text-[1rem]"
        style={{ animation: "reveal 600ms 220ms ease-out backwards" }}
      >
        <Step n="i" body="Type a prompt in §I and click generate." />
        <Step
          n="ii"
          body="The dashboard lights up. §III shows which attention heads attended to what; §IV shows what the model would have said at each layer."
        />
        <Step
          n="iii"
          body="In §VI, derive a steering direction from contrastive prompts and watch the α-slider push generation along it."
        />
        <Step
          n="iv"
          body="§VII is the headline: six methods for ablating refusal, scored against a probe that detects the model's internal sense of harm. When refusal-rate collapses but the probe holds, the method only muted what the model says — not what it knows."
        />
      </ol>

      <p
        className="mt-6 font-serif text-xs italic text-slate-500"
        style={{ animation: "reveal 600ms 360ms ease-out backwards" }}
      >
        <span className="not-italic mr-1 text-slate-700">·</span>
        Terms with a{" "}
        <span className="not-italic font-display align-super text-[0.7em] text-slate-400">
          †
        </span>{" "}
        carry a footnote — hover, focus, or tap to read it.{" "}
        <a
          href="https://github.com/lymnal/clearbox_ai/tree/proposal/research-direction-2025/docs/lessons"
          target="_blank"
          rel="noreferrer noopener"
          className="not-italic font-mono text-cerulean-light underline decoration-dotted underline-offset-4 hover:text-cerulean"
        >
          full curriculum
        </a>{" "}
        for the curious.{" "}
        <button
          type="button"
          onClick={hide}
          className="ml-2 not-italic font-display underline decoration-dotted underline-offset-4 text-slate-500 hover:text-vermillion-light"
        >
          hide intro
        </button>
      </p>
    </aside>
  );
}

/**
 * States the result up front.
 *
 * The bench table in §VII is six rows of deltas and confidence intervals. A
 * reader who already works on refusal directions can read the conclusion out
 * of it; nobody else can, and the conclusion is the most interesting thing on
 * the page. So it is written down, with the caveat that makes it honest —
 * n=20 gives intervals wide enough that "no technique passed" is partly a
 * statement about the power of the test.
 */
function FindingCallout() {
  return (
    <div className="mt-7 border-l-2 border-vermillion/60 py-1 pl-5">
      <p className="label-caps text-vermillion">The result</p>
      <p className="mt-2 font-serif text-[1.02rem] leading-[1.65] text-slate-300">
        Across all six techniques on Llama-3.2-1B,{" "}
        <strong className="font-display font-normal text-graphite">
          none met the preregistered bar for removing the model&apos;s internal
          sense of harm.
        </strong>{" "}
        The shape of the failure is the interesting part. The only two
        techniques whose harm signal demonstrably survived — Cheng and Herring,
        where a permutation test rejects chance at p&nbsp;=&nbsp;.025 and .008 —
        are precisely the two that <em>never reduced refusal at all</em>. The
        four that did collapse refusal left too little signal to measure either
        way.
      </p>
      <p className="mt-3 font-serif text-[1.02rem] leading-[1.65] text-slate-300">
        So this run does not show that ablation is a fluent gag. It shows
        something narrower and, for anyone planning the next experiment, more
        useful: at this sample size the test cannot tell the two apart. For all
        four refusal-collapsing methods the post-ablation AUC sits above chance
        by point estimate (0.60–0.72) and the interval runs down past it.
      </p>
      <p className="mt-3 font-serif text-sm italic leading-relaxed text-slate-500">
        n=20 pairs per class — 5 eval prompts, 10 AUC points. Bootstrap CIs are
        correspondingly wide, so &ldquo;no technique passed&rdquo; is in part a
        statement about the power of the test; a higher-power run is the obvious
        next step. Arditi and COSMIC resolve to the same direction here and are
        not independent evidence.
      </p>
    </div>
  );
}

function Step({ n, body }: { n: string; body: string }) {
  return (
    <li className="flex items-baseline gap-4">
      <span className="font-display text-vermillion-light flex-shrink-0 w-6 not-italic">
        {n}.
      </span>
      <span>{body}</span>
    </li>
  );
}
