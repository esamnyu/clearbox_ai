import { useEffect, useState } from "react";
import { useAnalysisStore } from "@/store/analysisStore";

interface SteeringPanelProps {
  prompt: string;
}

const NUM_LAYERS = 12;

export default function SteeringPanel({ prompt }: SteeringPanelProps) {
  const backendStatus = useAnalysisStore((s) => s.backendStatus);
  const backendError = useAnalysisStore((s) => s.backendError);
  const contrastivePairs = useAnalysisStore((s) => s.contrastivePairs);
  const steeringVector = useAnalysisStore((s) => s.steeringVector);
  const steeredResult = useAnalysisStore((s) => s.steeredResult);
  const steeringLayer = useAnalysisStore((s) => s.steeringLayer);
  const steeringAlpha = useAnalysisStore((s) => s.steeringAlpha);

  const fetchContrastivePairs = useAnalysisStore(
    (s) => s.fetchContrastivePairs,
  );
  const computeSteeringVector = useAnalysisStore(
    (s) => s.computeSteeringVector,
  );
  const runSteeredGeneration = useAnalysisStore((s) => s.runSteeredGeneration);
  const setSteeringLayer = useAnalysisStore((s) => s.setSteeringLayer);
  const setSteeringAlpha = useAnalysisStore((s) => s.setSteeringAlpha);

  const [isComputingVector, setIsComputingVector] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  // Fetch contrastive pairs on mount when backend is connected
  useEffect(() => {
    if (backendStatus === "connected" && contrastivePairs === null) {
      fetchContrastivePairs();
    }
  }, [backendStatus, contrastivePairs, fetchContrastivePairs]);

  const handleComputeVector = async () => {
    if (contrastivePairs === null) return;

    const positivePrompts = contrastivePairs.pairs.map((p) => p.positive);
    const negativePrompts = contrastivePairs.pairs.map((p) => p.negative);

    setIsComputingVector(true);
    try {
      await computeSteeringVector(
        positivePrompts,
        negativePrompts,
        steeringLayer,
      );
    } finally {
      setIsComputingVector(false);
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setIsGenerating(true);
    try {
      await runSteeredGeneration(prompt);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <section className="p-6 bg-slate-900/50 rounded-xl border border-slate-800">
      <h2 className="text-lg font-semibold text-white mb-4">
        Activation Steering
      </h2>

      {/* Backend disconnected warning */}
      {backendStatus !== "connected" && (
        <p className="text-sm text-amber-400 mb-4">
          Python backend is not connected. Start the backend server to use
          activation steering.
        </p>
      )}

      {/* Backend error */}
      {backendError && (
        <p className="text-sm text-red-400 mb-4">{backendError}</p>
      )}

      {/* Section 1: Contrastive Pairs */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-slate-300 mb-2">
          Contrastive Pairs
        </h3>

        {contrastivePairs === null ? (
          <p className="text-sm text-slate-500 italic">Loading pairs...</p>
        ) : (
          <div className="space-y-1 max-h-48 overflow-auto">
            {contrastivePairs.pairs.map((pair, index) => (
              <div key={index} className="flex gap-4 text-xs font-mono">
                <span className="text-green-400 flex-1 truncate">
                  {pair.positive}
                </span>
                <span className="text-red-400 flex-1 truncate">
                  {pair.negative}
                </span>
              </div>
            ))}
            <p className="text-xs text-slate-500 mt-1">
              {contrastivePairs.count} pairs total
            </p>
          </div>
        )}
      </div>

      {/* Section 2: Compute Vector */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-slate-300 mb-2">
          Compute Vector
        </h3>

        <div className="flex items-center gap-3 mb-3">
          <label className="flex items-center gap-2 text-sm text-slate-400">
            Layer
            <select
              value={steeringLayer}
              onChange={(e) => setSteeringLayer(Number(e.target.value))}
              className="bg-slate-950 border border-slate-700 text-slate-200 text-sm rounded-md px-2 py-1 focus:ring-2 focus:ring-blue-500 outline-none"
            >
              {Array.from({ length: NUM_LAYERS }, (_, i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            onClick={handleComputeVector}
            disabled={
              isComputingVector ||
              contrastivePairs === null ||
              backendStatus !== "connected"
            }
            className="px-4 py-1.5 text-sm font-medium rounded-md bg-purple-600 text-white hover:bg-purple-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {isComputingVector ? "Computing..." : "Compute Steering Vector"}
          </button>
        </div>

        {isComputingVector && (
          <p className="text-sm text-slate-400 italic">
            Computing steering vector...
          </p>
        )}

        {!isComputingVector && steeringVector !== null && (
          <div className="grid grid-cols-3 gap-3 text-sm">
            <div className="bg-slate-950 rounded-md px-3 py-2 border border-slate-800">
              <span className="text-slate-500 block text-xs">Vector Norm</span>
              <span className="text-slate-200 font-mono">
                {steeringVector.vector_norm.toFixed(4)}
              </span>
            </div>
            <div className="bg-slate-950 rounded-md px-3 py-2 border border-slate-800">
              <span className="text-slate-500 block text-xs">Layer</span>
              <span className="text-slate-200 font-mono">
                {steeringVector.layer}
              </span>
            </div>
            <div className="bg-slate-950 rounded-md px-3 py-2 border border-slate-800">
              <span className="text-slate-500 block text-xs">Pairs</span>
              <span className="text-slate-200 font-mono">
                +{steeringVector.n_positive} / -{steeringVector.n_negative}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Section 3: Steered Generation */}
      {steeringVector !== null && (
        <div>
          <h3 className="text-sm font-medium text-slate-300 mb-2">
            Steered Generation
          </h3>

          {/* Alpha slider */}
          <div className="mb-3">
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-slate-400">
                Alpha:{" "}
                <span className="font-mono text-slate-200">
                  {steeringAlpha.toFixed(1)}
                </span>
              </span>
              <span className="text-xs text-slate-500">
                Negative = negative sentiment, Positive = positive sentiment
              </span>
            </div>
            <input
              type="range"
              min={-3}
              max={3}
              step={0.1}
              value={steeringAlpha}
              onChange={(e) => setSteeringAlpha(Number(e.target.value))}
              className="w-full accent-purple-500"
            />
            <div className="flex justify-between text-xs text-slate-600">
              <span>-3.0</span>
              <span>0</span>
              <span>+3.0</span>
            </div>
          </div>

          <button
            type="button"
            onClick={handleGenerate}
            disabled={
              isGenerating || !prompt.trim() || backendStatus !== "connected"
            }
            className="px-4 py-1.5 text-sm font-medium rounded-md bg-purple-600 text-white hover:bg-purple-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors mb-4"
          >
            {isGenerating ? "Generating..." : "Generate with Steering"}
          </button>

          {isGenerating && (
            <p className="text-sm text-slate-400 italic">
              Running steered generation...
            </p>
          )}

          {/* Side-by-side results */}
          {!isGenerating && steeredResult !== null && (
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="bg-slate-950 rounded-md p-4 border border-slate-800">
                <h4 className="text-xs font-medium text-slate-400 mb-2">
                  Baseline
                </h4>
                <p className="font-mono text-sm text-white whitespace-pre-wrap">
                  {steeredResult.baseline_text}
                </p>
              </div>
              <div className="bg-slate-950 rounded-md p-4 border border-purple-700/50">
                <h4 className="text-xs font-medium text-purple-400 mb-2">
                  Steered (&alpha;={steeredResult.alpha})
                </h4>
                <p className="font-mono text-sm text-purple-200 whitespace-pre-wrap">
                  {steeredResult.steered_text}
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
