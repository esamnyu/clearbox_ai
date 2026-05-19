import { Fragment, useMemo } from "react";
import { useAnalysisStore } from "@/store/analysisStore";
import type { HeadGridCell } from "@/store/analysisStore";

const NUM_LAYERS = 12;
const NUM_HEADS = 12;

const PATTERN_COLORS: Record<HeadGridCell["pattern"], string> = {
  previous_token: "bg-cerulean",
  first_token: "bg-emerald-500",
  induction: "bg-vermillion",
  broadcasting: "bg-amber-500",
  other: "bg-slate-700",
};

const PATTERN_LABELS: Record<HeadGridCell["pattern"], string> = {
  previous_token: "previous token",
  first_token: "first token",
  induction: "induction",
  broadcasting: "broadcasting",
  other: "other",
};

function entropyToOpacity(
  entropy: number,
  minEntropy: number,
  maxEntropy: number,
): number {
  if (maxEntropy === minEntropy) return 1.0;
  const normalized = (entropy - minEntropy) / (maxEntropy - minEntropy);
  return 1.0 - normalized * 0.7;
}

function buildTooltip(cell: HeadGridCell): string {
  const lines = [
    `Layer ${cell.layer}, Head ${cell.head}`,
    cell.pattern.replace("_", " "),
    `entropy ${cell.entropy.toFixed(2)}`,
  ];
  if (cell.isInduction)
    lines.push(`induction ${cell.inductionScore.toFixed(2)}`);
  return lines.join(" · ");
}

export default function HeadGrid() {
  const headGrid = useAnalysisStore((s) => s.headGrid);
  const selectedLayer = useAnalysisStore((s) => s.selectedLayer);
  const selectedHead = useAnalysisStore((s) => s.selectedHead);
  const setSelectedLayer = useAnalysisStore((s) => s.setSelectedLayer);
  const setSelectedHead = useAnalysisStore((s) => s.setSelectedHead);

  const { minEntropy, maxEntropy } = useMemo(() => {
    if (headGrid === null) return { minEntropy: 0, maxEntropy: 1 };
    let min = Infinity;
    let max = -Infinity;
    for (const layerRow of headGrid) {
      for (const cell of layerRow) {
        if (cell.entropy < min) min = cell.entropy;
        if (cell.entropy > max) max = cell.entropy;
      }
    }
    return { minEntropy: min, maxEntropy: max };
  }, [headGrid]);

  return (
    <div className="relative overflow-hidden border border-rule bg-paper">
      <div className="atmosphere" />
      <div className="relative p-8 sm:p-10">
        <header className="mb-6">
          <h3 className="label-caps text-graphite">Head Classification</h3>
          <p className="mt-1 font-serif text-xs italic text-slate-500">
            12 × 12 grid of attention heads, color-classified by pattern;
            opacity ∝ inverse entropy
          </p>
        </header>

        {headGrid === null ? (
          <p className="font-serif italic text-slate-700">
            run inference with attention capture to view head classifications
          </p>
        ) : (
          <>
            <div className="overflow-auto pb-2">
              <div
                className="inline-grid gap-px"
                style={{
                  gridTemplateColumns: `auto repeat(${NUM_HEADS}, 30px)`,
                  gridTemplateRows: `auto repeat(${NUM_LAYERS}, 30px)`,
                }}
              >
                <div />
                {Array.from({ length: NUM_HEADS }, (_, h) => (
                  <div
                    key={`col-${h}`}
                    className="flex items-end justify-center pb-1 font-mono text-[0.6rem] text-slate-500"
                  >
                    {h.toString().padStart(2, "0")}
                  </div>
                ))}
                {headGrid.map((layerRow, l) => (
                  <Fragment key={`row-${l}`}>
                    <div className="flex items-center justify-end whitespace-nowrap pr-2 font-mono text-[0.7rem] text-slate-400">
                      L{l.toString().padStart(2, "0")}
                    </div>
                    {layerRow.map((cell) => {
                      const isSelected =
                        cell.layer === selectedLayer &&
                        cell.head === selectedHead;
                      const opacity = entropyToOpacity(
                        cell.entropy,
                        minEntropy,
                        maxEntropy,
                      );
                      return (
                        <button
                          key={`cell-${cell.layer}-${cell.head}`}
                          type="button"
                          className={
                            "relative flex h-[30px] w-[30px] cursor-pointer items-center justify-center transition-shadow " +
                            PATTERN_COLORS[cell.pattern] +
                            (isSelected
                              ? " ring-1 ring-graphite ring-offset-1 ring-offset-paper"
                              : "")
                          }
                          style={{ opacity }}
                          title={buildTooltip(cell)}
                          onClick={() => {
                            setSelectedLayer(cell.layer);
                            setSelectedHead(cell.head);
                          }}
                        >
                          {cell.isInduction && (
                            <span className="select-none text-[10px] leading-none text-white/95">
                              ★
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </Fragment>
                ))}
              </div>
            </div>

            <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-rule pt-4">
              <span className="label-caps text-slate-500">legend</span>
              {(
                Object.keys(PATTERN_COLORS) as Array<HeadGridCell["pattern"]>
              ).map((pattern) => (
                <div key={pattern} className="flex items-center gap-1.5">
                  <span
                    className={
                      "inline-block h-2.5 w-2.5 " + PATTERN_COLORS[pattern]
                    }
                  />
                  <span className="font-serif text-xs italic text-slate-400">
                    {PATTERN_LABELS[pattern]}
                  </span>
                </div>
              ))}
              <span className="ml-auto font-serif text-xs italic text-slate-500">
                ★ marks induction heads
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
