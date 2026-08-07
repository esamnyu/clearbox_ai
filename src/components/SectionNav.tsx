import { useEffect, useState } from "react";

export interface NavSection {
  id: string;
  number: string;
  label: string;
  /** Why this section can't be used yet. Undefined means it's usable now. */
  blockedBy?: string;
}

/**
 * Sticky rail across the top of the workbench.
 *
 * The page is seven long sections deep and previously had no anchors at all, so
 * the only way to reach the headline result in §VII was to scroll past
 * everything else and hope. This gives the reader a map, a sense of place, and
 * — via `blockedBy` — an answer to "why is that panel empty?" before they have
 * to guess.
 */
export default function SectionNav({ sections }: { sections: NavSection[] }) {
  const [active, setActive] = useState<string>(sections[0]?.id ?? "");

  // Order is stable and cheap to derive; keeping it as a primitive string means
  // the effect below doesn't re-subscribe on every parent render just because
  // `sections` is a fresh array literal each time (the blockedBy fields change
  // as the model loads).
  const order = sections.map((s) => s.id).join(",");

  useEffect(() => {
    const ids = order.split(",").filter(Boolean);
    const nodes = ids
      .map((id) => document.getElementById(id))
      .filter((n): n is HTMLElement => n !== null);
    if (nodes.length === 0) return;

    // A callback only receives the entries whose intersection *changed*, so
    // deciding from `entries` alone loses track of everything already on
    // screen — the active item then sticks on whichever section happened to
    // report last. Keep the live set here and pick from it in document order.
    const onScreen = new Set<string>();

    // rootMargin pulls the trigger band to roughly the top third of the
    // viewport: a heading should read as "current" once it is comfortably in
    // view, not the instant its first pixel crosses the fold.
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) onScreen.add(e.target.id);
          else onScreen.delete(e.target.id);
        }
        const first = ids.find((id) => onScreen.has(id));
        if (first) setActive(first);
      },
      { rootMargin: "-15% 0px -60% 0px", threshold: 0 },
    );

    nodes.forEach((n) => observer.observe(n));
    return () => observer.disconnect();
  }, [order]);

  const go = (id: string) => {
    const el = document.getElementById(id);
    if (!el) return;
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    el.scrollIntoView({
      behavior: reduced ? "auto" : "smooth",
      block: "start",
    });
    // Move focus for keyboard and screen-reader users; scrollIntoView alone
    // leaves the caret behind at the link.
    el.setAttribute("tabindex", "-1");
    el.focus({ preventScroll: true });
  };

  return (
    <nav
      aria-label="Workbench sections"
      className="sticky top-0 z-30 -mx-6 mb-12 border-b border-slate-800/60 bg-ink/85 px-6 backdrop-blur"
    >
      <ul className="flex snap-x items-stretch gap-1 overflow-x-auto py-2 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {sections.map((s) => {
          const isActive = active === s.id;
          const blocked = Boolean(s.blockedBy);
          return (
            <li key={s.id} className="snap-start">
              <button
                type="button"
                onClick={() => go(s.id)}
                aria-current={isActive ? "location" : undefined}
                title={s.blockedBy ? `${s.label} — ${s.blockedBy}` : s.label}
                className={[
                  "group flex items-baseline gap-2 whitespace-nowrap rounded px-3 py-1.5",
                  "font-display text-sm transition-colors",
                  "focus-visible:outline focus-visible:outline-1 focus-visible:outline-vermillion",
                  isActive
                    ? "bg-vermillion/10 text-vermillion"
                    : "text-slate-500 hover:text-slate-300",
                ].join(" ")}
              >
                <span className="italic">{s.number}.</span>
                <span>{s.label}</span>
                {blocked && (
                  <span
                    aria-hidden
                    className="text-[0.6rem] leading-none text-slate-600"
                  >
                    ●
                  </span>
                )}
                <span className="sr-only">
                  {blocked ? ` — ${s.blockedBy}` : " — ready"}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
