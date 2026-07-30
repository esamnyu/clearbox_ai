import { useMemo, useState } from "react";
import { useAnalysisStore } from "@/store/analysisStore";
import type { LogitLensResponse } from "@/lib/api";

interface LogitLensProps {
  prompt: string;
}

const TOP_K_DISPLAY = 5;

function getFinalPrediction(
  predictions: LogitLensResponse["predictions"],
): string | null {
  if (predictions.length === 0) return null;
  const lastRow = predictions[predictions.length - 1];
  if (lastRow.top_k.length === 0) return null;
  return lastRow.top_k[0].token;
}

function formatLayerLabel(layer: string): string {
  if (layer === "Embed") return "embed";
  const m = layer.match(/^Layer\s+(\d+)$/);
  return m ? `L${m[1].padStart(2, "0")}` : layer;
}

export default function LogitLens({ prompt }: LogitLensProps) {
  const backendStatus = useAnalysisStore((s) => s.backendStatus);
  const backendError = useAnalysisStore((s) => s.backendError);
  const logitLensResult = useAnalysisStore((s) => s.logitLensResult);
  const runLogitLens = useAnalysisStore((s) => s.runLogitLens);

  const [isLoading, setIsLoading] = useState(false);

  const finalToken = useMemo(
    () =>
      logitLensResult ? getFinalPrediction(logitLensResult.predictions) : null,
    [logitLensResult],
  );

  const handleRun = async () => {
    setIsLoading(true);
    try {
      await runLogitLens(prompt);
    } finally {
      setIsLoading(false);
    }
  };

  const backendOk = backendStatus === "connected";

  return (
    <div className="relative overflow-hidden border border-rule bg-paper">
      <div className="atmosphere" />
      <div className="relative p-8 sm:p-10">
        <header className="mb-6 flex items-baseline justify-between gap-4">
          <div>
            <h3 className="label-caps text-graphite">Logit Lens</h3>
            <p className="mt-1 font-serif text-xs italic text-slate-500">
              what the model would predict, layer by layer
            </p>
          </div>
          <button
            type="button"
            disabled={isLoading || !backendOk || !prompt.trim()}
            onClick={handleRun}
            className="group inline-flex items-center gap-2 font-display text-sm italic text-cerulean-light transition-colors hover:text-cerulean-light/90 disabled:cursor-not-allowed disabled:text-slate-700"
          >
            <span className="text-base leading-none" aria-hidden>
              ↪
            </span>
            <span className="border-b border-dotted border-cerulean/40 pb-px group-hover:border-cerulean-light group-disabled:border-slate-800">
              {isLoading ? "reading…" : "run logit lens"}
            </span>
          </button>
        </header>

        {!backendOk && (
          <p className="mb-6 font-serif italic text-sm text-vermillion-light">
            backend unreachable
          </p>
        )}
        {backendError && (
          <p className="mb-6 font-serif italic text-sm text-vermillion-light">
            {backendError}
          </p>
        )}

        {isLoading && (
          <p className="font-serif italic text-slate-500">
            ▍ projecting residuals through unembedding…
          </p>
        )}

        {!isLoading && logitLensResult && (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-rule">
                  <th className="label-caps pb-3 pr-4 text-left text-slate-500">
                    layer
                  </th>
                  {Array.from({ length: TOP_K_DISPLAY }, (_, i) => (
                    <th
                      key={i}
                      className="label-caps pb-3 pl-3 pr-2 text-left text-slate-500"
                    >
                      #{i + 1}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {logitLensResult.predictions.map((row, rowIdx) => (
                  <tr
                    key={rowIdx}
                    className="border-b border-rule/40 last:border-b-0"
                  >
                    <td className="py-2 pr-4 font-mono text-xs text-slate-400 align-middle whitespace-nowrap">
                      {formatLayerLabel(row.layer)}
                    </td>
                    {Array.from({ length: TOP_K_DISPLAY }, (_, k) => {
                      const p = row.top_k[k] ?? null;
                      if (!p) return <td key={k} className="px-3 py-2" />;

                      const isFinalMatch =
                        finalToken !== null && p.token === finalToken;
                      const intensity = Math.max(0.06, p.prob);

                      return (
                        <td key={k} className="px-3 py-2">
                          <div
                            className={
                              "inline-flex flex-col items-start rounded-sm px-2.5 py-1 " +
                              (isFinalMatch ? "ring-1 ring-vermillion/70" : "")
                            }
                            style={{
                              backgroundColor: `rgba(56, 116, 156, ${intensity})`,
                            }}
                            title={`${p.token}: ${(p.prob * 100).toFixed(1)}%`}
                          >
                            <span className="font-mono text-sm leading-tight text-graphite">
                              {p.token.replace(/^Ġ/, "·")}
                            </span>
                            <span className="font-mono text-[0.65rem] leading-tight tabular-nums text-slate-400">
                              {(p.prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            {finalToken && (
              <p className="mt-4 font-serif text-xs italic text-slate-500">
                <span className="inline-block h-1.5 w-1.5 -translate-y-px rounded-full bg-vermillion mr-2 align-middle" />
                vermillion ring marks the layer&apos;s match with the final
                prediction
                <span className="not-italic font-mono text-graphite">
                  {" "}
                  &nbsp;{finalToken.replace(/^Ġ/, "·")}
                </span>
              </p>
            )}
          </div>
        )}

        {!isLoading && !logitLensResult && backendOk && (
          <p className="font-serif italic text-slate-700">
            run the lens to see predictions stratified by layer
          </p>
        )}
      </div>
    </div>
  );
}
