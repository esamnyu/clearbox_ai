import { Fragment, useMemo } from "react";
import { useAnalysisStore } from "@/store/analysisStore";

const NUM_LAYERS = 12;
const NUM_HEADS = 12;

/**
 * Build a 2D attention matrix for a single head from a 3D TensorView.
 *
 * The layer tensor has shape [num_heads, seq_len, seq_len].
 * We extract head `h` by iterating over (i, j) and calling `.get(h, i, j)`.
 */
function extractHeadMatrix(
  layerTensor: {
    get: (...indices: number[]) => number;
    shape: readonly number[];
  },
  head: number,
): number[][] {
  const seqLen = layerTensor.shape[1];
  const matrix: number[][] = [];

  for (let i = 0; i < seqLen; i++) {
    const row: number[] = [];
    for (let j = 0; j < seqLen; j++) {
      row.push(layerTensor.get(head, i, j));
    }
    matrix.push(row);
  }

  return matrix;
}

export default function AttentionHeatmap() {
  const attentions = useAnalysisStore((s) => s.attentions);
  const tokens = useAnalysisStore((s) => s.tokens);
  const selectedLayer = useAnalysisStore((s) => s.selectedLayer);
  const selectedHead = useAnalysisStore((s) => s.selectedHead);
  const setSelectedLayer = useAnalysisStore((s) => s.setSelectedLayer);
  const setSelectedHead = useAnalysisStore((s) => s.setSelectedHead);

  const matrix = useMemo(() => {
    if (attentions === null) return null;

    const layerTensor = attentions.get(selectedLayer);
    if (!layerTensor) return null;

    return extractHeadMatrix(layerTensor, selectedHead);
  }, [attentions, selectedLayer, selectedHead]);

  return (
    <section className="p-6 bg-slate-900/50 rounded-xl border border-slate-800">
      <h2 className="text-lg font-semibold text-white mb-4">
        Attention Pattern
      </h2>

      {/* Layer / Head selectors */}
      <div className="flex gap-4 mb-6">
        <label className="flex items-center gap-2 text-sm text-slate-400">
          Layer
          <select
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(Number(e.target.value))}
            className="bg-slate-950 border border-slate-700 text-slate-200 text-sm rounded-md px-2 py-1 focus:ring-2 focus:ring-blue-500 outline-none"
          >
            {Array.from({ length: NUM_LAYERS }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2 text-sm text-slate-400">
          Head
          <select
            value={selectedHead}
            onChange={(e) => setSelectedHead(Number(e.target.value))}
            className="bg-slate-950 border border-slate-700 text-slate-200 text-sm rounded-md px-2 py-1 focus:ring-2 focus:ring-blue-500 outline-none"
          >
            {Array.from({ length: NUM_HEADS }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Content */}
      {attentions === null || matrix === null ? (
        <p className="text-sm text-slate-500 italic">
          Run inference with attention capture to view the heatmap.
        </p>
      ) : (
        <div className="overflow-auto">
          {/* Grid wrapper: row labels column + heatmap columns */}
          <div
            className="inline-grid gap-px"
            style={{
              gridTemplateColumns: `auto repeat(${tokens.length}, minmax(28px, 1fr))`,
              gridTemplateRows: `auto repeat(${tokens.length}, minmax(28px, 1fr))`,
            }}
          >
            {/* Top-left empty corner cell */}
            <div />

            {/* Column headers (key tokens — "attending TO") */}
            {tokens.map((token, j) => (
              <div
                key={`col-${j}`}
                className="text-xs font-mono text-slate-400 flex items-end justify-center pb-1"
                style={{
                  writingMode: "vertical-rl",
                  transform: "rotate(180deg)",
                  height: "60px",
                }}
              >
                <span className="truncate max-w-[56px]">{token}</span>
              </div>
            ))}

            {/* Rows */}
            {matrix.map((row, i) => (
              <Fragment key={`row-${i}`}>
                {/* Row label (query token — "attending FROM") */}
                <div className="text-xs font-mono text-slate-400 flex items-center justify-end pr-2 whitespace-nowrap">
                  <span className="truncate max-w-[72px]">{tokens[i]}</span>
                </div>

                {/* Heatmap cells */}
                {row.map((weight, j) => (
                  <div
                    key={`cell-${i}-${j}`}
                    className="min-w-[28px] min-h-[28px] rounded-sm"
                    style={{
                      backgroundColor: `rgba(59, 130, 246, ${weight})`,
                    }}
                    title={`[${i}, ${j}] ${weight.toFixed(4)}`}
                  />
                ))}
              </Fragment>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
