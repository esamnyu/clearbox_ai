import { useId, useState, type ReactNode } from "react";

interface DefinedTermProps {
  /** The term as it appears in the header (e.g. "Refusal Bench"). */
  children: ReactNode;
  /** Plain-English definition. Keep to ~2 sentences. */
  definition: string;
  /**
   * Optional href to a deeper lesson, e.g. "docs/lessons/06-harmfulness-probe.md".
   * Rendered as "→ Lesson N" in vermillion when supplied.
   */
  lessonHref?: string;
  /** Optional display label for the lesson link. Defaults to "→ read more". */
  lessonLabel?: string;
}

/**
 * Inline footnote-style defined term.
 *
 * Visual grammar:
 *   <term> dotted-underlined, followed by a slate-500 superscript dagger.
 *   On hover / focus / click, a paper-tone popover fades in beneath the term
 *   with the definition in italic Newsreader and an optional vermillion
 *   "→ Lesson N" link in mono.
 *
 * Designed to read as a marginalia footnote in a manuscript, not as a SaaS
 * "?" help-icon tooltip. Compose freely inside <h1>..<h4>, button labels,
 * table headers, etc.
 *
 * Accessibility:
 *   - Button semantics so it's keyboard-focusable
 *   - aria-describedby connects the trigger to the popover id
 *   - aria-expanded toggles on click for touch / mobile users
 */
export default function DefinedTerm({
  children,
  definition,
  lessonHref,
  lessonLabel = "→ read more",
}: DefinedTermProps) {
  const popoverId = useId();
  const [pinned, setPinned] = useState(false);

  return (
    <span className="defined-term group relative inline-block">
      <button
        type="button"
        aria-describedby={popoverId}
        aria-expanded={pinned}
        onClick={() => setPinned((p) => !p)}
        className="defined-term__trigger relative inline cursor-help bg-transparent p-0 font-[inherit] text-[inherit] outline-none"
      >
        <span className="defined-term__label">{children}</span>
        <span
          aria-hidden
          className="defined-term__marker ml-0.5 align-super text-[0.55em] font-display tracking-normal text-slate-500"
        >
          †
        </span>
      </button>

      <span
        id={popoverId}
        role="tooltip"
        data-pinned={pinned ? "true" : undefined}
        className="defined-term__popover"
      >
        <span className="defined-term__def">{definition}</span>
        {lessonHref ? (
          <a
            href={lessonHref}
            target="_blank"
            rel="noreferrer noopener"
            className="defined-term__lesson"
          >
            {lessonLabel}
          </a>
        ) : null}
      </span>
    </span>
  );
}
