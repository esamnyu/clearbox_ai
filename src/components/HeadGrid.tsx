import { Fragment, useMemo } from "react";
import { useAnalysisStore } from "@/store/analysisStore";
import type { HeadGridCell } from "@/store/analysisStore";

const NUM_LAYERS = 12;
const NUM_HEADS = 12;

const PATTERN_COLORS: Record<HeadGridCell["pattern"], string> = {
  previous_token: "bg-blue-500",
  first_token: "bg-green-500",
  induction: "bg-orange-500",
  broadcasting: "bg-slate-500",
  other: "bg-slate-700",
};

const PATTERN_LABELS: Record<HeadGridCell["pattern"], string> = {
  previous_token: "Previous Token",
  first_token: "First Token",
  induction: "Induction",
  broadcasting: "Broadcasting",
  other: "Other",
};

/**
 * Map entropy to opacity in [0.3, 1.0].
 * Lower entropy (more focused) = higher opacity (more visible).
 */
function entropyToOpacity(
  entropy: number,
  minEntropy: number,
  maxEntropy: number,
): number {
  if (maxEntropy === minEntropy) return 1.0;

  // Normalized 0..1 where 0 = min entropy, 1 = max entropy
  const normalized = (entropy - minEntropy) / (maxEntropy - minEntropy);

  // Invert: low entropy -> high opacity
  // Map to [0.3, 1.0]
  return 1.0 - normalized * 0.7;
}

function buildTooltip(cell: HeadGridCell): string {
  const lines = [
    `Layer ${cell.layer}, Head ${cell.head}`,
    cell.pattern,
    `Entropy: ${cell.entropy.toFixed(2)}`,
  ];

  if (cell.isInduction) {
    lines.push(`Induction: ${cell.inductionScore.toFixed(2)}`);
  }

  return lines.join("\n");
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
    <section className="p-6 bg-slate-900/50 rounded-xl border border-slate-800">
      <h2 className="text-lg font-semibold text-white mb-4">
        Head Classification
      </h2>

      {headGrid === null ? (
        <p className="text-sm text-slate-500 italic">
          Run inference with attention capture to view head classifications.
        </p>
      ) : (
        <>
          <div className="overflow-auto">
            <div
              className="inline-grid gap-px"
              style={{
                gridTemplateColumns: `auto repeat(${NUM_HEADS}, 32px)`,
                gridTemplateRows: `auto repeat(${NUM_LAYERS}, 32px)`,
              }}
            >
              {/* Top-left empty corner */}
              <div />

              {/* Column headers: H0..H11 */}
              {Array.from({ length: NUM_HEADS }, (_, h) => (
                <div
                  key={`col-${h}`}
                  className="text-xs text-slate-500 font-mono flex items-end justify-center pb-1"
                >
                  H{h}
                </div>
              ))}

              {/* Grid rows */}
              {headGrid.map((layerRow, l) => (
                <Fragment key={`row-${l}`}>
                  {/* Row label */}
                  <div className="text-xs text-slate-500 font-mono flex items-center justify-end pr-2 whitespace-nowrap">
                    Layer {l}
                  </div>

                  {/* Head cells */}
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
                        className={`
                          w-8 h-8 rounded-sm relative flex items-center justify-center
                          cursor-pointer transition-shadow
                          ${PATTERN_COLORS[cell.pattern]}
                          ${isSelected ? "ring-2 ring-white" : ""}
                        `}
                        style={{ opacity }}
                        title={buildTooltip(cell)}
                        onClick={() => {
                          setSelectedLayer(cell.layer);
                          setSelectedHead(cell.head);
                        }}
                      >
                        {cell.isInduction && (
                          <span className="text-[10px] leading-none text-white/90 select-none">
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

          {/* Legend */}
          <div className="flex flex-wrap gap-4 mt-4">
            {(
              Object.keys(PATTERN_COLORS) as Array<HeadGridCell["pattern"]>
            ).map((pattern) => (
              <div key={pattern} className="flex items-center gap-1.5">
                <span
                  className={`inline-block w-3 h-3 rounded-full ${PATTERN_COLORS[pattern]}`}
                />
                <span className="text-xs text-slate-400">
                  {PATTERN_LABELS[pattern]}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
