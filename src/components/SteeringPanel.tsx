import { useEffect, useState } from "react";
import { useAnalysisStore } from "@/store/analysisStore";
import AblationHero from "@/components/AblationHero";

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
  const ablatedResult = useAnalysisStore((s) => s.ablatedResult);
  const runAblation = useAnalysisStore((s) => s.runAblation);

  const [isComputingVector, setIsComputingVector] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isAblating, setIsAblating] = useState(false);

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

  const handleAblate = async () => {
    if (!prompt.trim()) return;
    setIsAblating(true);
    try {
      await runAblation(prompt);
    } finally {
      setIsAblating(false);
    }
  };

  const backendOk = backendStatus === "connected";

  return (
    <div className="relative overflow-hidden border border-rule bg-paper">
      <div className="atmosphere" />
      <div className="relative p-8 sm:p-10">
        {!backendOk && (
          <p className="mb-6 font-serif italic text-sm text-vermillion-light">
            backend unreachable — start it locally or point VITE_API_BASE at a
            deployed instance.
          </p>
        )}
        {backendError && (
          <p className="mb-6 font-serif italic text-sm text-vermillion-light">
            {backendError}
          </p>
        )}

        {/* ─── ACT I — Contrastive Pairs ─────────────────── */}
        <Movement label="Pairs" caption="contrastive prompts">
          {contrastivePairs === null ? (
            <p className="font-serif italic text-slate-700">loading pairs…</p>
          ) : (
            <div className="overflow-hidden">
              <table className="w-full font-mono text-[0.825rem]">
                <thead>
                  <tr className="border-b border-rule">
                    <th className="label-caps pb-2 pr-4 text-left text-cerulean">
                      positive
                    </th>
                    <th className="label-caps pb-2 text-left text-vermillion">
                      negative
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-rule/60">
                  {contrastivePairs.pairs.map((pair, i) => (
                    <tr key={i}>
                      <td className="py-1.5 pr-6 text-cerulean-light">
                        {pair.positive}
                      </td>
                      <td className="py-1.5 text-vermillion-light">
                        {pair.negative}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-3 font-serif text-xs italic text-slate-500">
                {contrastivePairs.count} pairs · validated to tokenize to
                identical lengths
              </p>
            </div>
          )}
        </Movement>

        {/* ─── ACT II — Compute Direction ─────────────────── */}
        <Movement label="Direction" caption="difference of means">
          <div className="flex flex-wrap items-center gap-x-8 gap-y-4">
            <EditorialNumberPicker
              label="extract at layer"
              value={steeringLayer}
              onChange={setSteeringLayer}
              max={NUM_LAYERS - 1}
            />
            <EditorialButton
              onClick={handleComputeVector}
              disabled={
                isComputingVector || contrastivePairs === null || !backendOk
              }
              accent="cerulean"
            >
              {isComputingVector ? "extracting…" : "extract direction"}
            </EditorialButton>
          </div>

          {steeringVector !== null && !isComputingVector && (
            <dl className="mt-6 grid grid-cols-3 gap-x-8 gap-y-1 border-t border-rule pt-4 font-mono text-sm">
              <Cell label="‖d‖" value={steeringVector.vector_norm.toFixed(3)} />
              <Cell label="layer" value={String(steeringVector.layer)} />
              <Cell
                label="pairs"
                value={`+${steeringVector.n_positive} / −${steeringVector.n_negative}`}
              />
            </dl>
          )}
        </Movement>

        {/* ─── ACT III — Steered Generation ───────────────── */}
        {steeringVector !== null && (
          <Movement label="Steered Generation" caption="h ← h + αd̂">
            <AlphaSlider value={steeringAlpha} onChange={setSteeringAlpha} />

            <EditorialButton
              onClick={handleGenerate}
              disabled={isGenerating || !prompt.trim() || !backendOk}
              accent="cerulean"
            >
              {isGenerating ? "generating…" : "generate with steering"}
            </EditorialButton>

            {steeredResult !== null && !isGenerating && (
              <div className="mt-8 grid grid-cols-1 gap-x-10 md:grid-cols-2">
                <ResultColumn
                  label="Baseline"
                  accent="slate"
                  text={steeredResult.baseline_text}
                />
                <ResultColumn
                  label={`Steered · α = ${steeredResult.alpha.toFixed(1)}`}
                  accent="cerulean"
                  text={steeredResult.steered_text}
                  divider
                />
              </div>
            )}
          </Movement>
        )}

        {/* ─── ACT IV — Ablation Hero ─────────────────────── */}
        {steeringVector !== null && (
          <AblationHero
            prompt={prompt}
            layer={steeringLayer}
            pairsCount={contrastivePairs?.count ?? 0}
            result={ablatedResult}
            isAblating={isAblating}
            onAblate={handleAblate}
            steeringVectorNorm={steeringVector?.vector_norm ?? null}
            disabled={!backendOk}
          />
        )}
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────
// Movement — a labeled act within the panel
// ──────────────────────────────────────────────────────────────

function Movement({
  label,
  caption,
  children,
}: {
  label: string;
  caption: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-10">
      <header className="mb-5 flex items-baseline gap-4">
        <h3 className="label-caps text-graphite">{label}</h3>
        <span className="font-serif text-xs italic text-slate-500">
          {caption}
        </span>
        <span className="rule-flex" />
      </header>
      {children}
    </section>
  );
}

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="label-caps text-slate-500">{label}</dt>
      <dd className="mt-1 text-graphite tabular-nums">{value}</dd>
    </div>
  );
}

function EditorialNumberPicker({
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
          className="appearance-none border-b border-dotted border-rule bg-transparent px-1 pr-5 py-0.5 font-display text-lg text-graphite focus:border-vermillion focus:outline-none"
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

function EditorialButton({
  onClick,
  disabled,
  accent,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  accent: "cerulean" | "vermillion";
  children: React.ReactNode;
}) {
  const cls =
    accent === "cerulean"
      ? "text-cerulean-light border-cerulean/40 hover:text-cerulean-light"
      : "text-vermillion-light border-vermillion/40 hover:text-vermillion-light";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`group inline-flex items-center gap-2 font-display text-sm italic transition-colors disabled:cursor-not-allowed disabled:text-slate-700 ${cls}`}
    >
      <span className="text-base leading-none" aria-hidden>
        ↪
      </span>
      <span className="border-b border-dotted pb-px group-disabled:border-slate-800">
        {children}
      </span>
    </button>
  );
}

function AlphaSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (n: number) => void;
}) {
  return (
    <div className="mb-6">
      <div className="mb-3 flex items-baseline justify-between">
        <span className="font-serif text-sm italic text-slate-500">
          alpha ={" "}
          <span className="font-mono not-italic text-graphite tabular-nums">
            {value.toFixed(1)}
          </span>
        </span>
        <span className="label-caps text-slate-600">
          <span className="text-vermillion">negative</span>
          <span className="px-2 text-slate-700">·</span>
          <span className="text-cerulean">positive</span>
        </span>
      </div>
      <div className="relative">
        <div
          className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2"
          style={{
            background:
              "linear-gradient(to right, #bd4931 0%, #23252f 50%, #38749c 100%)",
          }}
          aria-hidden
        />
        <input
          type="range"
          min={-3}
          max={3}
          step={0.1}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="relative w-full appearance-none bg-transparent accent-vermillion-light [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-1 [&::-webkit-slider-thumb]:bg-graphite [&::-webkit-slider-thumb]:cursor-pointer"
          style={{ height: "1rem" }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[0.65rem] text-slate-600">
        <span>−3.0</span>
        <span>0</span>
        <span>+3.0</span>
      </div>
    </div>
  );
}

function ResultColumn({
  label,
  accent,
  text,
  divider,
}: {
  label: string;
  accent: "slate" | "cerulean";
  text: string;
  divider?: boolean;
}) {
  const accentClass =
    accent === "cerulean" ? "text-cerulean" : "text-slate-400";
  const cleaned = text.replace(/^<\|endoftext\|>/, "").trimStart();
  return (
    <div
      className={
        "relative " +
        (divider
          ? "mt-8 md:mt-0 md:border-l md:border-dotted md:border-rule md:pl-10"
          : "")
      }
    >
      <div className="mb-4 flex items-center gap-3">
        <span className="rule-flex" />
        <span className={"label-caps " + accentClass}>{label}</span>
        <span className="rule-flex" />
      </div>
      <p className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-graphite">
        {cleaned}
      </p>
    </div>
  );
}
