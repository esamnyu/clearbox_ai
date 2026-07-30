import { useState } from "react";
import { useAnalysisStore } from "@/store/analysisStore";

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

  const backendOk = backendStatus === "connected";

  // Top-3 tokens for the inline summary line.
  const topThree = gradientResult
    ? [...gradientResult.gradient_norms]
        .sort((a, b) => b.normalized - a.normalized)
        .slice(0, 3)
    : null;

  return (
    <div className="relative overflow-hidden border border-rule bg-paper">
      <div className="atmosphere" />
      <div className="relative p-8 sm:p-10">
        <header className="mb-6">
          <h3 className="label-caps text-graphite">Token Importance</h3>
          <p className="mt-1 font-serif text-xs italic text-slate-500">
            ∂(target probability)/∂(input embedding) — L2 norm per position
          </p>
        </header>

        {!backendOk && (
          <p className="mb-6 font-serif italic text-sm text-vermillion-light">
            backend unreachable
          </p>
        )}

        <div className="mb-6 flex flex-wrap items-baseline gap-x-4 gap-y-3 border-b border-rule pb-5">
          <label className="label-caps text-slate-500">target token</label>
          <input
            type="text"
            value={targetToken}
            onChange={(e) => setTargetToken(e.target.value)}
            placeholder="Paris"
            className="flex-1 max-w-xs border-b border-dotted border-rule bg-transparent px-1 py-1 font-mono text-base text-graphite placeholder:italic placeholder:text-slate-700 focus:border-vermillion focus:outline-none"
          />
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={
              isLoading || !targetToken.trim() || !prompt.trim() || !backendOk
            }
            className="group inline-flex items-center gap-2 font-display text-sm italic text-vermillion-light transition-colors hover:text-vermillion-light/90 disabled:cursor-not-allowed disabled:text-slate-700"
          >
            <span className="text-base leading-none" aria-hidden>
              ↪
            </span>
            <span className="border-b border-dotted border-vermillion/40 pb-px group-hover:border-vermillion-light group-disabled:border-slate-800">
              {isLoading ? "computing…" : "compute gradients"}
            </span>
          </button>
        </div>

        {backendError && (
          <p className="mb-4 font-serif italic text-sm text-vermillion-light">
            {backendError}
          </p>
        )}

        {isLoading && (
          <p className="font-serif italic text-slate-500">
            ▍ backpropagating through 12 layers…
          </p>
        )}

        {!isLoading && gradientResult !== null && (
          <>
            <div className="flex flex-wrap gap-x-2 gap-y-2">
              {gradientResult.gradient_norms.map((entry, i) => {
                const heat = entry.normalized;
                const tone = Math.max(0.06, heat);
                return (
                  <span
                    key={i}
                    className="relative inline-flex items-center font-mono text-base"
                    style={{
                      color: heat > 0.55 ? "#ffffff" : "#9da3b8",
                    }}
                    title={`${entry.token} · ∂L=${entry.norm.toFixed(4)}`}
                  >
                    <span
                      className="absolute inset-0 -z-[1] rounded-sm"
                      style={{
                        background: `rgba(189, 73, 49, ${tone})`,
                      }}
                      aria-hidden
                    />
                    <span className="relative px-1.5 py-0.5">
                      {entry.token.replace(/^Ġ/, " ").replace(/^Ċ/, "⏎")}
                    </span>
                  </span>
                );
              })}
            </div>

            {topThree && topThree.length > 0 && (
              <p className="mt-6 border-t border-rule pt-4 font-serif text-xs italic text-slate-500">
                most susceptible to perturbation:{" "}
                {topThree.map((t, i) => (
                  <span key={i}>
                    {i > 0 && (
                      <span className="not-italic text-slate-700"> · </span>
                    )}
                    <span className="not-italic font-mono text-vermillion-light">
                      {t.token.replace(/^Ġ/, "·")}
                    </span>{" "}
                    <span className="not-italic font-mono text-slate-500 tabular-nums">
                      ({t.normalized.toFixed(2)})
                    </span>
                  </span>
                ))}
              </p>
            )}
          </>
        )}

        {!isLoading &&
          gradientResult === null &&
          !backendError &&
          backendOk && (
            <p className="font-serif italic text-slate-700">
              enter a target token to see which input positions most influence
              it
            </p>
          )}
      </div>
    </div>
  );
}
