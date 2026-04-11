import { useMemo, useState } from "react";
import { useAnalysisStore } from "@/store/analysisStore";
import type { LogitLensResponse } from "@/lib/api";

interface LogitLensProps {
  prompt: string;
}

/** Number of top-k predictions shown per layer column. */
const TOP_K_DISPLAY = 5;

/**
 * Derive the final-layer top prediction token so we can highlight convergence
 * in earlier layers. Returns null when no data is available.
 */
function getFinalPrediction(
  predictions: LogitLensResponse["predictions"],
): string | null {
  if (predictions.length === 0) return null;
  const lastRow = predictions[predictions.length - 1];
  if (lastRow.top_k.length === 0) return null;
  return lastRow.top_k[0].token;
}

/**
 * Format a layer label for the row header.
 * "Embed" stays as-is; "Layer N" becomes "LN".
 */
function formatLayerLabel(layer: string): string {
  if (layer === "Embed") return "Embed";
  const match = layer.match(/^Layer\s+(\d+)$/);
  return match ? `L${match[1]}` : layer;
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

  return (
    <section className="p-6 bg-slate-900/50 rounded-xl border border-slate-800">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">Logit Lens</h2>

        <button
          type="button"
          disabled={
            isLoading || backendStatus !== "connected" || !prompt.trim()
          }
          onClick={handleRun}
          className="px-3 py-1.5 text-sm font-medium rounded-md bg-violet-600 hover:bg-violet-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? "Analyzing..." : "Run Logit Lens"}
        </button>
      </div>

      {/* Backend disconnected warning */}
      {backendStatus !== "connected" && (
        <p className="text-sm text-amber-400 mb-4">
          Start the Python backend to use logit lens analysis
        </p>
      )}

      {/* Backend error */}
      {backendError && (
        <p className="text-sm text-red-400 mb-4">{backendError}</p>
      )}

      {/* Loading spinner */}
      {isLoading && (
        <div className="flex items-center gap-2 text-sm text-slate-400 mb-4">
          <svg
            className="animate-spin h-4 w-4 text-violet-400"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          Analyzing...
        </div>
      )}

      {/* Results table */}
      {!isLoading && logitLensResult && (
        <div className="overflow-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="text-left font-mono text-xs text-slate-400 pb-2 pr-4 whitespace-nowrap">
                  Layer
                </th>
                {Array.from({ length: TOP_K_DISPLAY }, (_, i) => (
                  <th
                    key={i}
                    className="text-left font-mono text-xs text-slate-500 pb-2 px-2 whitespace-nowrap"
                  >
                    #{i + 1}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {logitLensResult.predictions.map((row, rowIdx) => (
                <tr key={rowIdx} className="border-t border-slate-800/50">
                  {/* Layer label */}
                  <td className="font-mono text-xs text-slate-400 py-1.5 pr-4 whitespace-nowrap align-top">
                    {formatLayerLabel(row.layer)}
                  </td>

                  {/* Top-k prediction cells */}
                  {Array.from({ length: TOP_K_DISPLAY }, (_, k) => {
                    const prediction = row.top_k[k] ?? null;
                    if (!prediction) {
                      return <td key={k} className="px-2 py-1.5" />;
                    }

                    const isFinalMatch =
                      finalToken !== null && prediction.token === finalToken;

                    return (
                      <td key={k} className="px-2 py-1.5">
                        <div
                          className={`inline-flex flex-col items-start px-2 py-1 rounded ${
                            isFinalMatch
                              ? "border border-green-500/50"
                              : "border border-transparent"
                          }`}
                          style={{
                            backgroundColor: `rgba(139, 92, 246, ${prediction.prob})`,
                          }}
                          title={`${prediction.token}: ${(prediction.prob * 100).toFixed(1)}%`}
                        >
                          <span className="font-mono text-sm text-white leading-tight">
                            {prediction.token}
                          </span>
                          <span className="text-xs text-slate-400 leading-tight">
                            {(prediction.prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !logitLensResult && backendStatus === "connected" && (
        <p className="text-sm text-slate-500 italic">
          Enter a prompt and run logit lens to see layer-by-layer predictions.
        </p>
      )}
    </section>
  );
}
