import { useState } from "react";
import { useAnalysisStore } from "@/store/analysisStore";
// GradientsResponse type available from @/lib/api if needed for explicit typing

interface TokenImportanceProps {
  prompt: string;
}

export default function TokenImportance({ prompt }: TokenImportanceProps) {
  const [targetToken, setTargetToken] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const gradientResult = useAnalysisStore((s) => s.gradientResult);
  const backendStatus = useAnalysisStore((s) => s.backendStatus);
  const backendError = useAnalysisStore((s) => s.backendError);
  const runGradients = useAnalysisStore((s) => s.runGradients);

  const handleAnalyze = async () => {
    if (!targetToken.trim() || !prompt.trim()) return;

    setIsLoading(true);
    try {
      await runGradients(prompt, targetToken.trim());
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section className="p-6 bg-slate-900/50 rounded-xl border border-slate-800">
      <h2 className="text-lg font-semibold text-white mb-4">
        Token Importance (Gradients)
      </h2>

      {backendStatus === "disconnected" && (
        <p className="text-sm text-amber-400 mb-4">
          Python backend is not connected. Start the backend server to use
          gradient analysis.
        </p>
      )}

      {/* Target token input */}
      <div className="flex items-center gap-3 mb-4">
        <label className="text-sm text-slate-400 shrink-0">Target token</label>
        <input
          type="text"
          value={targetToken}
          onChange={(e) => setTargetToken(e.target.value)}
          placeholder='e.g. "Paris"'
          className="bg-slate-950 border border-slate-700 text-slate-200 text-sm rounded-md px-3 py-1.5 focus:ring-2 focus:ring-blue-500 outline-none flex-1 max-w-xs"
        />
        <button
          type="button"
          onClick={handleAnalyze}
          disabled={isLoading || !targetToken.trim() || !prompt.trim()}
          className="px-4 py-1.5 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? "Computing..." : "Analyze Gradients"}
        </button>
      </div>

      {/* Error display */}
      {backendError && (
        <p className="text-sm text-red-400 mb-4">{backendError}</p>
      )}

      {/* Loading state */}
      {isLoading && (
        <p className="text-sm text-slate-400 italic">Computing gradients...</p>
      )}

      {/* Token chips */}
      {!isLoading && gradientResult !== null && (
        <div className="flex flex-wrap gap-1.5 mt-2">
          {gradientResult.gradient_norms.map((entry, index) => (
            <span
              key={index}
              className="font-mono text-sm px-2 py-0.5 rounded"
              style={{
                backgroundColor: `rgba(239, 68, 68, ${entry.normalized})`,
                color: entry.normalized > 0.5 ? "white" : "#e2e8f0",
              }}
              title={`${entry.token}\nGradient: ${entry.norm.toFixed(4)}\nNormalized: ${entry.normalized.toFixed(4)}`}
            >
              {entry.token}
            </span>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && gradientResult === null && !backendError && (
        <p className="text-sm text-slate-500 italic">
          Enter a target token and click Analyze to view gradient-based token
          importance.
        </p>
      )}
    </section>
  );
}
