import { useEffect, useState } from "react";
import type { AblationResponse, NextTokenTopK } from "@/lib/api";

interface AblationHeroProps {
  prompt: string;
  layer: number;
  pairsCount: number;
  result: AblationResponse | null;
  isAblating: boolean;
  onAblate: () => void;
  steeringVectorNorm: number | null;
  disabled?: boolean;
}

export default function AblationHero({
  prompt,
  layer,
  pairsCount,
  result,
  isAblating,
  onAblate,
  steeringVectorNorm,
  disabled,
}: AblationHeroProps) {
  // Forces CSS animations to re-trigger on each new result without relying on
  // a transition library. Keys on Column + math underline change → React
  // remounts those nodes → animations replay.
  const [revealKey, setRevealKey] = useState(0);
  useEffect(() => {
    if (result) setRevealKey((k) => k + 1);
  }, [result]);

  const promptDisplay =
    prompt.length > 38 ? `"${prompt.slice(0, 38)}…"` : `"${prompt}"`;

  return (
    <section className="relative isolate -mx-6 -mb-6 mt-10 overflow-hidden rounded-b-xl border-t border-rule bg-ink px-6 pb-10 pt-9 sm:px-10">
      {/* atmospheric depth — two soft radials in the semantic palette */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.06] mix-blend-screen"
        style={{
          backgroundImage:
            "radial-gradient(circle at 18% -10%, rgba(189,73,49,0.55), transparent 42%), radial-gradient(circle at 100% 110%, rgba(56,116,156,0.45), transparent 38%)",
        }}
      />
      {/* film-grain noise, very subtle */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.025] mix-blend-overlay"
        style={{
          backgroundImage:
            "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='160' height='160' filter='url(%23n)' opacity='0.9'/></svg>\")",
        }}
      />

      {/* HEADER BAND */}
      <header className="relative mb-12 flex items-baseline justify-between">
        <h3 className="font-display text-[0.7rem] uppercase tracking-[0.34em] text-slate-400">
          Ablate <span className="px-1 text-slate-700">·</span> Direction
        </h3>
        <div className="flex items-baseline gap-3 text-[0.65rem] uppercase tracking-[0.34em] text-slate-500">
          <span>Layer</span>
          <span
            className="font-display text-base font-light normal-case tracking-normal text-slate-100"
            style={{ fontVariationSettings: '"opsz" 144' }}
          >
            {layer.toString().padStart(2, "0")}
          </span>
        </div>
      </header>

      {/* THE MATH — visual centerpiece */}
      <figure className="relative mb-14 text-center">
        <MathDisplay revealKey={revealKey} active={isAblating} />
        <figcaption className="mt-7 inline-flex items-center gap-3 font-serif text-[0.75rem] italic text-vermillion">
          <span
            key={`fc-l-${revealKey}`}
            className="block h-px w-8 origin-right bg-vermillion/40 animate-rule-grow"
            style={{ animationDelay: "260ms" }}
          />
          removed
          <span
            key={`fc-r-${revealKey}`}
            className="block h-px w-8 origin-left bg-vermillion/40 animate-rule-grow"
            style={{ animationDelay: "260ms" }}
          />
        </figcaption>
      </figure>

      {/* TWO-COLUMN BODY — manuscript spread */}
      <div className="relative grid grid-cols-1 gap-x-12 md:grid-cols-2">
        <Column
          key={`base-${revealKey}`}
          label="Baseline"
          accent="cerulean"
          text={result?.baseline_text}
          topk={result?.next_token_topk?.baseline}
          loading={isAblating}
          revealDelay={0}
        />
        <Column
          key={`abl-${revealKey}`}
          label="Ablated"
          accent="vermillion"
          text={result?.ablated_text}
          topk={result?.next_token_topk?.ablated}
          loading={isAblating}
          revealDelay={320}
          divider
        />
      </div>

      {/* FOOTER BAND — meta + action */}
      <footer className="relative mt-12 flex flex-wrap items-baseline justify-between gap-4 border-t border-dashed border-rule pt-5">
        <div className="font-serif text-xs italic text-slate-500">
          {steeringVectorNorm !== null ? (
            <>
              <span className="font-mono not-italic text-slate-300">
                ‖<span className="italic">d̂</span>‖ ={" "}
                {steeringVectorNorm.toFixed(3)}
              </span>
              <Sep />
              <span>
                {pairsCount} sentiment pair{pairsCount === 1 ? "" : "s"}
              </span>
              <Sep />
              <span>
                prompt:{" "}
                <span className="font-mono not-italic text-slate-300">
                  {promptDisplay}
                </span>
              </span>
            </>
          ) : (
            <span>compute a steering vector above to enable ablation</span>
          )}
        </div>

        <button
          type="button"
          onClick={onAblate}
          disabled={isAblating || disabled || !prompt.trim()}
          className="group inline-flex items-center gap-2 font-display text-sm tracking-wide text-vermillion-light transition-colors hover:text-vermillion-light/90 disabled:cursor-not-allowed disabled:text-slate-700"
        >
          <span className="text-base leading-none" aria-hidden>
            ↪
          </span>
          <span
            className={
              "border-b border-dotted border-vermillion/40 pb-px italic group-hover:border-vermillion-light " +
              "group-disabled:border-slate-800"
            }
          >
            {isAblating
              ? "running ablation…"
              : result
                ? "compute again"
                : "ablate"}
          </span>
        </button>
      </footer>
    </section>
  );
}

function Sep() {
  return <span className="mx-3 text-slate-700">·</span>;
}

function MathDisplay({
  revealKey,
  active,
}: {
  revealKey: number;
  active: boolean;
}) {
  return (
    <div
      className={
        "font-display text-[2rem] leading-none text-slate-100 sm:text-[2.5rem] md:text-[2.75rem] " +
        (active ? "animate-math-glow" : "")
      }
      style={{ fontVariationSettings: '"opsz" 144, "SOFT" 0' }}
    >
      <span className="italic">h</span>
      <span className="ml-0.5 align-top text-[0.55em] italic text-slate-300">
        ′
      </span>
      <span className="mx-5 text-slate-500">=</span>
      <span className="italic">h</span>
      <span className="mx-4 text-slate-400">−</span>
      <span className="relative inline-block">
        <span className="text-slate-400">(</span>
        <span className="italic">h</span>
        <span className="mx-1.5 text-slate-400">·</span>
        <DHat />
        <span className="text-slate-400">)</span>
        <DHat />
        <span
          key={`underline-${revealKey}`}
          aria-hidden
          className="pointer-events-none absolute -bottom-2 left-0 right-0 h-px origin-center bg-vermillion animate-rule-grow"
          style={{ animationDelay: "180ms" }}
        />
      </span>
    </div>
  );
}

// Render d̂ via a precisely-positioned overlay rather than relying on
// the U+0302 combining diacritic, which renders inconsistently across
// fonts/browsers at large display sizes.
function DHat() {
  return (
    <span className="relative inline-block italic">
      d
      <span
        aria-hidden
        className="pointer-events-none absolute left-1/2 not-italic"
        style={{
          top: "-0.32em",
          transform: "translateX(-46%)",
          fontSize: "0.62em",
          letterSpacing: 0,
          lineHeight: 1,
        }}
      >
        ̂
      </span>
    </span>
  );
}

interface ColumnProps {
  label: string;
  accent: "vermillion" | "cerulean";
  text: string | undefined;
  topk: NextTokenTopK["baseline"] | undefined;
  loading: boolean;
  revealDelay: number;
  divider?: boolean;
}

function Column({
  label,
  accent,
  text,
  topk,
  loading,
  revealDelay,
  divider,
}: ColumnProps) {
  const accentClass =
    accent === "vermillion" ? "text-vermillion" : "text-cerulean";
  const accentBg = accent === "vermillion" ? "bg-vermillion" : "bg-cerulean";

  return (
    <div
      className={
        "relative " +
        (divider
          ? "md:border-l md:border-dotted md:border-rule md:pl-12 mt-10 md:mt-0"
          : "")
      }
    >
      {/* small-caps label flanked by hairlines — manuscript section break */}
      <div className="mb-5 flex items-center gap-3">
        <span
          className="h-px flex-1 bg-rule"
          style={{
            animation: `rule-grow 520ms ${revealDelay}ms ease-out backwards`,
            transformOrigin: "right",
          }}
        />
        <span
          className={
            "font-display text-[0.65rem] uppercase tracking-[0.4em] " +
            accentClass
          }
          style={{
            animation: `reveal 480ms ${revealDelay}ms ease-out backwards`,
          }}
        >
          {label}
        </span>
        <span
          className="h-px flex-1 bg-rule"
          style={{
            animation: `rule-grow 520ms ${revealDelay}ms ease-out backwards`,
            transformOrigin: "left",
          }}
        />
      </div>

      {/* generation body */}
      <div
        className="min-h-[6rem] font-mono text-sm leading-relaxed text-slate-100"
        style={{
          animation: text
            ? `reveal 520ms ${revealDelay + 80}ms ease-out backwards`
            : undefined,
        }}
      >
        {loading && !text ? (
          <span className="inline-block text-slate-600 animate-pulse">
            ▍ <span className="italic font-serif">thinking</span>
          </span>
        ) : text ? (
          <Generation text={text} />
        ) : (
          <span className="text-slate-700 italic font-serif">
            run an ablation to see the generation
          </span>
        )}
      </div>

      {/* δ-logprob bars — only rendered when backend supplies them */}
      {topk && topk.length > 0 ? (
        <div className="mt-7 space-y-2">
          <div className="mb-2 flex items-baseline gap-2 font-serif text-[0.65rem] italic text-slate-500">
            <span>next token</span>
            <span className="block h-px flex-1 bg-rule" />
            <span className="font-mono not-italic">
              top {Math.min(5, topk.length)}
            </span>
          </div>
          {topk.slice(0, 5).map((row, i) => (
            <div
              key={`${label}-${i}-${row.token}`}
              className="grid grid-cols-[minmax(0,1fr)_3.5rem_3rem] items-center gap-3 text-xs"
              style={{
                animation: `reveal 380ms ${revealDelay + 200 + i * 50}ms ease-out backwards`,
              }}
            >
              <div className="flex items-center gap-2 min-w-0">
                <span
                  className={"block h-[2px] " + accentBg}
                  style={{
                    width: `${Math.max(3, row.prob * 100)}%`,
                    minWidth: "3%",
                  }}
                />
              </div>
              <span className="truncate font-mono text-slate-200">
                {displayToken(row.token)}
              </span>
              <span className="text-right font-mono text-slate-500 tabular-nums">
                {row.prob.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function Generation({ text }: { text: string }) {
  // Strip GPT-2's BOS token if the model echoed it; show whitespace honestly.
  const cleaned = text.replace(/^<\|endoftext\|>/, "").trimStart();
  return <span className="whitespace-pre-wrap">{cleaned}</span>;
}

function displayToken(t: string): string {
  // GPT-2 BPE: leading space encoded as 'Ġ', newline as 'Ċ'. Render visibly.
  return t.replace(/^Ġ/, "·").replace(/Ċ/g, "⏎").replace(/^\s$/, "·");
}
