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
        n interpretability workbench for the smaller open language models — a
        place to watch what happens inside a transformer, then change one thing
        and watch again. Six published methods for stripping a model of its
        refusal are wired here side-by-side; one of them, almost certainly, is a
        fluent gag rather than a genuine intervention. Below, what the page
        expects of you.
      </p>

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
