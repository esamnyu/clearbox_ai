import { Fragment, useMemo } from "react";
import { useAnalysisStore } from "@/store/analysisStore";

const NUM_LAYERS = 12;
const NUM_HEADS = 12;

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
    <div className="relative overflow-hidden border border-rule bg-paper">
      <div className="atmosphere" />
      <div className="relative p-8 sm:p-10">
        <header className="mb-6">
          <h3 className="label-caps text-graphite">Attention Pattern</h3>
          <p className="mt-1 font-serif text-xs italic text-slate-500">
            who looks at whom · single layer, single head
          </p>
        </header>

        <div className="mb-7 flex flex-wrap gap-x-6 gap-y-3">
          <Selector
            label="layer"
            value={selectedLayer}
            onChange={setSelectedLayer}
            max={NUM_LAYERS - 1}
          />
          <Selector
            label="head"
            value={selectedHead}
            onChange={setSelectedHead}
            max={NUM_HEADS - 1}
          />
        </div>

        {attentions === null || matrix === null ? (
          <p className="font-serif italic text-slate-700">
            run inference with attention capture to view the heatmap
          </p>
        ) : (
          <div className="overflow-auto">
            <div
              className="inline-grid gap-px"
              style={{
                gridTemplateColumns: `auto repeat(${tokens.length}, minmax(26px, 1fr))`,
                gridTemplateRows: `auto repeat(${tokens.length}, minmax(26px, 1fr))`,
              }}
            >
              <div />
              {tokens.map((token, j) => (
                <div
                  key={`col-${j}`}
                  className="flex items-end justify-center pb-1 font-mono text-[0.65rem] text-slate-400"
                  style={{
                    writingMode: "vertical-rl",
                    transform: "rotate(180deg)",
                    height: "64px",
                  }}
                >
                  <span className="max-w-[56px] truncate">{token}</span>
                </div>
              ))}
              {matrix.map((row, i) => (
                <Fragment key={`row-${i}`}>
                  <div className="flex items-center justify-end whitespace-nowrap pr-3 font-mono text-[0.7rem] text-slate-400">
                    <span className="max-w-[72px] truncate">{tokens[i]}</span>
                  </div>
                  {row.map((weight, j) => (
                    <div
                      key={`cell-${i}-${j}`}
                      className="min-h-[26px] min-w-[26px]"
                      style={{
                        backgroundColor: `rgba(56, 116, 156, ${Math.max(0.05, weight)})`,
                      }}
                      title={`[${i}, ${j}] ${weight.toFixed(4)}`}
                    />
                  ))}
                </Fragment>
              ))}
            </div>
            <p className="mt-5 font-serif text-xs italic text-slate-500">
              rows: querying token · columns: attended-to token · intensity ∝
              weight
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function Selector({
  label,
  value,
  onChange,
  max,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  max: number;
}) {
  return (
    <label className="inline-flex items-baseline gap-3">
      <span className="label-caps text-slate-500">{label}</span>
      <span className="relative">
        <select
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="appearance-none border-b border-dotted border-rule bg-transparent px-1 pr-5 py-0.5 font-display text-base text-graphite focus:border-vermillion focus:outline-none"
          style={{ fontVariationSettings: '"opsz" 144' }}
        >
          {Array.from({ length: max + 1 }, (_, i) => (
            <option key={i} value={i} className="bg-paper text-graphite">
              {i}
            </option>
          ))}
        </select>
        <span
          className="pointer-events-none absolute right-0 top-1/2 -translate-y-1/2 text-xs text-slate-500"
          aria-hidden
        >
          ▾
        </span>
      </span>
    </label>
  );
}
