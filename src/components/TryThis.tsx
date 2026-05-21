interface TryThisProps {
  /** The suggested action, e.g. "type 'The Eiffel Tower is located in', then generate". */
  hint: string;
  /**
   * Optional outcome, the editorial whisper after the action.
   * Rendered in slate-500.
   */
  outcome?: string;
  /** Tailwind className, e.g. for vertical spacing. */
  className?: string;
}

/**
 * Marginalia line under a section header.
 *
 * Format:
 *   ¶  Try: <hint>.  <outcome>.
 *
 * Reads as an editor's whispered suggestion, not as a tutorial step.
 * Italic Newsreader throughout; the hint sits in graphite, the outcome
 * in slate-500. The pilcrow is the only ornament.
 */
export default function TryThis({
  hint,
  outcome,
  className = "",
}: TryThisProps) {
  return (
    <p
      className={
        "font-serif text-[0.95rem] italic leading-relaxed text-slate-300 " +
        className
      }
    >
      <span
        aria-hidden
        className="not-italic mr-2 text-slate-600"
        style={{ fontFeatureSettings: '"ss01"' }}
      >
        ¶
      </span>
      <span className="text-graphite">Try:</span>{" "}
      <span className="text-slate-300">{hint}</span>
      {outcome ? (
        <>
          {" "}
          <span className="not-italic mx-1 text-slate-700">·</span>{" "}
          <span className="text-slate-500">{outcome}</span>
        </>
      ) : null}
    </p>
  );
}
