import { useEffect, useState, type ReactNode } from "react";
import { useModelStore } from "./store/modelStore";
import { useAnalysisStore } from "./store/analysisStore";
import AttentionHeatmap from "./components/AttentionHeatmap";
import HeadGrid from "./components/HeadGrid";
import LogitLens from "./components/LogitLens";
import TokenImportance from "./components/TokenImportance";
import SteeringPanel from "./components/SteeringPanel";
import RefusalBenchLeaderboard from "./components/RefusalBenchLeaderboard";
import WelcomeDeck from "./components/WelcomeDeck";
import DefinedTerm from "./components/DefinedTerm";
import TryThis from "./components/TryThis";
import { getRefusalPairs } from "./lib/api";
import type { TensorWithMetadata } from "./engine/types";

// GitHub URL for the durable lessons; lessonHref on DefinedTerm jumps the
// curious reader to the relevant chapter without leaving the manuscript.
const LESSON_BASE =
  "https://github.com/lymnal/clearbox_ai/blob/proposal/research-direction-2025/docs/lessons";

const ALL_BENCH_TECHNIQUES = [
  "arditi",
  "cheng",
  "cosmic",
  "herring",
  "maskey",
  "wollschlager",
];

interface TelemetryData {
  tokenIds: number[];
  attentions?: TensorWithMetadata[];
  hiddenStates?: TensorWithMetadata[];
}

function computeTelemetryStats(t: TelemetryData | null) {
  if (!t) return null;
  const aCount = t.attentions?.length ?? 0;
  const hCount = t.hiddenStates?.length ?? 0;
  const layers = new Set([
    ...(t.attentions?.map((x) => x.layer) ?? []),
    ...(t.hiddenStates?.map((x) => x.layer) ?? []),
  ]).size;
  const steps = new Set([
    ...(t.attentions?.map((x) => x.step) ?? []),
    ...(t.hiddenStates?.map((x) => x.step) ?? []),
  ]).size;
  const aMB =
    t.attentions?.reduce((s, x) => s + x.data.byteLength / 1048576, 0) ?? 0;
  const hMB =
    t.hiddenStates?.reduce((s, x) => s + x.data.byteLength / 1048576, 0) ?? 0;
  return {
    tokens: t.tokenIds?.length ?? 0,
    attn: aCount,
    hidden: hCount,
    layers,
    steps,
    totalMB: Number((aMB + hMB).toFixed(2)),
  };
}

export default function App() {
  const [prompt, setPrompt] = useState("The Eiffel Tower is located in");
  const [genText, setGenText] = useState("");
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const {
    status,
    loadProgress,
    initWorker,
    disposeWorker,
    loadModel,
    generate,
  } = useModelStore();
  const analyze = useAnalysisStore((s) => s.analyze);
  const checkBackend = useAnalysisStore((s) => s.checkBackend);
  const backendStatus = useAnalysisStore((s) => s.backendStatus);
  const steeringLayer = useAnalysisStore((s) => s.steeringLayer);

  const [refusalHarmful, setRefusalHarmful] = useState<string[]>([]);
  const [refusalHarmless, setRefusalHarmless] = useState<string[]>([]);

  useEffect(() => {
    initWorker();
    checkBackend();
    // Terminating on unmount is what makes this effect safe to run twice.
    // Without it, StrictMode's deliberate double-invoke left an orphaned
    // Worker per mount — each holding its own transformers.js instance, and
    // each able to fire onerror into shared store state after being abandoned.
    return () => disposeWorker();
  }, [initWorker, disposeWorker, checkBackend]);

  useEffect(() => {
    // Fetch refusal pairs once the backend is reachable. The endpoint
    // returns count=0 until backend/refusal_pairs.py is populated by
    // running backend/scripts/build_refusal_pairs.py locally.
    if (backendStatus !== "connected") return;
    if (refusalHarmful.length > 0 || refusalHarmless.length > 0) return;
    getRefusalPairs()
      .then((res) => {
        setRefusalHarmful(res.pairs.map((p) => p.harmful));
        setRefusalHarmless(res.pairs.map((p) => p.harmless));
      })
      .catch(() => {
        // Endpoint may 404 on older deploys or return empty before
        // population; surface the empty-state UI either way.
      });
  }, [backendStatus, refusalHarmful.length, refusalHarmless.length]);

  const handleGenerate = async () => {
    if (status !== "ready") return;
    setIsGenerating(true);
    setGenText("Generating…");
    setTelemetry(null);
    try {
      const result = await generate(prompt);
      setGenText(result.text);
      setTelemetry({
        tokenIds: result.tokenIds,
        attentions: result.attentions,
        hiddenStates: result.hiddenStates,
      });
      analyze(result);
    } catch (e) {
      console.error("Generation error:", e);
      setGenText("");
    } finally {
      setIsGenerating(false);
    }
  };

  const stats = computeTelemetryStats(telemetry);

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-ink font-serif text-graphite">
      <div className="atmosphere" />

      <main className="relative mx-auto max-w-[88rem] px-6 pb-32 pt-12 sm:px-10 lg:px-16">
        <Masthead
          modelStatus={status}
          loadProgress={loadProgress}
          backendStatus={backendStatus}
          onLoadModel={() => loadModel("Xenova/gpt2")}
        />

        <WelcomeDeck />

        <Section
          number="I"
          title="Prompt"
          tryThis={{
            hint: "type 'The Eiffel Tower is located in' and press generate",
            outcome:
              "everything downstream — attention heads, layer-by-layer predictions, ablations — keys off this one sentence",
          }}
        >
          <PromptInput value={prompt} onChange={setPrompt} />
          <PrimaryAction
            onClick={handleGenerate}
            disabled={status !== "ready" || isGenerating}
            loading={isGenerating}
          >
            {status !== "ready"
              ? "load the model to enable inference"
              : isGenerating
                ? "running inference"
                : "generate & analyze"}
          </PrimaryAction>
        </Section>

        {(genText || stats) && (
          <Section
            number="II"
            title="Generation"
            tryThis={{
              hint: "compare what came out with the telemetry on the right",
              outcome:
                "each attention tensor and hidden state captured here feeds the panels below",
            }}
          >
            <GenerationView text={genText} stats={stats} />
          </Section>
        )}

        <Section
          number="III"
          title={
            <DefinedTerm
              definition="How each token in your prompt 'looks at' the others to decide what to write next. A GPT-2 model has 12 layers × 12 heads = 144 of these patterns running in parallel; some specialise in copying, some in tracking position, some are full induction circuits."
              lessonHref={`${LESSON_BASE}/01-residual-streams.md`}
              lessonLabel="→ Lesson 01 · residual streams"
            >
              Attention
            </DefinedTerm>
          }
          tryThis={{
            hint: "click an orange-tagged cell in the head grid",
            outcome:
              "orange cells are induction heads — they implement in-context learning. The heatmap on the right reveals what they actually attend to.",
          }}
        >
          <div className="grid grid-cols-1 gap-10 xl:grid-cols-[1fr_1.15fr]">
            <HeadGrid />
            <AttentionHeatmap />
          </div>
        </Section>

        <Section
          number="IV"
          title={
            <DefinedTerm
              definition="What the model would predict next if you halted it at each layer. The trick: multiply the residual stream at layer L by the unembedding matrix, see what tokens come out top. Early layers usually say 'the'; later layers converge on the actual answer."
              lessonHref={`${LESSON_BASE}/01-residual-streams.md`}
              lessonLabel="→ Lesson 01 · residual streams"
            >
              Layer Predictions
            </DefinedTerm>
          }
          tryThis={{
            hint: "watch the prediction collapse from 'the' to 'Paris' between layers 9 and 11",
            outcome:
              "this is the moment the model 'figures it out'. Lewis Smith / nostalgebraist coined this view in 2020.",
          }}
        >
          <LogitLens prompt={prompt} />
        </Section>

        <Section
          number="V"
          title={
            <DefinedTerm
              definition="Which tokens in your prompt most influence a target prediction. Computed from the gradient of the target's log-probability with respect to each token's embedding. High-gradient tokens are the levers a clever attack would pull."
              lessonHref={`${LESSON_BASE}/03-direction-extraction.md`}
              lessonLabel="→ Lesson 03 · direction extraction"
            >
              Token Importance
            </DefinedTerm>
          }
          tryThis={{
            hint: "type the prompt, then 'Paris' as the target token",
            outcome:
              "'Eiffel', 'city', and 'Tower' light up — the French-flavored tokens. An adversarial attack would swap them out.",
          }}
        >
          <TokenImportance prompt={prompt} />
        </Section>

        <Section
          number="VI"
          title="Intervention"
          tryThis={{
            hint: "compute a steering vector, then slide α from −3 to +3 and watch generation tilt",
            outcome:
              "at α = +1.5 the model talks about sunsets; at α = −1.5 it complains about traffic. Same prompt.",
          }}
        >
          <SteeringPanel prompt={prompt} />
        </Section>

        <Section
          number="VII"
          title={
            <DefinedTerm
              definition="A head-to-head comparison of six published methods for stripping a safety-trained model of its refusal. The bench scores each method on two axes: does the model stop saying 'I refuse' (Δ refusal rate) and does its internal sense that the request is harmful actually disappear (Δ AUC, a probe-based measurement)."
              lessonHref={`${LESSON_BASE}/07-what-the-bench-measures.md`}
              lessonLabel="→ Lesson 07 · what the bench measures"
            >
              Refusal Bench
            </DefinedTerm>
          }
          tryThis={{
            // The rows below are a cached local run, not something the visitor
            // can reproduce here: the deployed backend is free-tier CPU, where
            // /refusal-bench times out on the heavier techniques (see
            // docs/DEPLOYMENT.md §1e). Promising "six rows render" from a click
            // was a claim the deploy cannot keep.
            hint: "read down the table",
            outcome:
              "where Δ refusal rate is big but Δ AUC is small, the method only suppressed speech, not understanding. 'Run bench' re-runs live against a local backend; on the free CPU deploy the heavier techniques time out.",
          }}
        >
          <RefusalBenchLeaderboard
            layer={steeringLayer}
            harmfulPrompts={refusalHarmful}
            harmlessPrompts={refusalHarmless}
            techniqueNames={ALL_BENCH_TECHNIQUES}
          />
        </Section>

        <Colophon />
      </main>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────
// Masthead — wordmark + tagline + connection status + load CTA
// ──────────────────────────────────────────────────────────────

interface MastheadProps {
  modelStatus: "idle" | "loading" | "ready" | "error";
  loadProgress: number;
  backendStatus: "unknown" | "connected" | "disconnected";
  onLoadModel: () => void;
}

function Masthead({
  modelStatus,
  loadProgress,
  backendStatus,
  onLoadModel,
}: MastheadProps) {
  return (
    <header className="mb-20 pb-8">
      <div className="flex flex-col items-start gap-2">
        <span className="label-caps text-vermillion">
          Mechanistic Interpretability
          <span className="px-2 text-slate-700">·</span>
          <span className="text-slate-500">An Open Workbench</span>
        </span>

        <h1
          className="font-display text-[3.25rem] leading-[0.95] tracking-[-0.01em] text-graphite sm:text-[4.5rem]"
          style={{ fontVariationSettings: '"opsz" 144, "SOFT" 30' }}
        >
          NeuroScope
          <span className="font-display italic text-vermillion">.</span>
        </h1>

        <p className="mt-3 max-w-2xl font-serif text-base italic leading-relaxed text-slate-400 sm:text-lg">
          A browser-resident reading room for GPT-2&apos;s internals — extract
          activations, project them out, watch the model change its mind.
        </p>
      </div>

      <div className="mt-8 flex flex-wrap items-center gap-x-8 gap-y-3 border-t border-rule pt-5">
        <StatusInline
          label="Browser model"
          status={
            modelStatus === "ready"
              ? "good"
              : modelStatus === "loading"
                ? "pending"
                : "down"
          }
          detail={
            modelStatus === "ready"
              ? "gpt2 · 124M · loaded"
              : modelStatus === "loading"
                ? `loading · ${loadProgress.toFixed(0)}%`
                : modelStatus === "error"
                  ? "load failed"
                  : "not loaded"
          }
        />
        <StatusInline
          label="Backend"
          status={
            backendStatus === "connected"
              ? "good"
              : backendStatus === "unknown"
                ? "pending"
                : "down"
          }
          detail={
            backendStatus === "connected"
              ? "transformerlens · ready"
              : backendStatus === "disconnected"
                ? "unreachable"
                : "checking…"
          }
        />

        {modelStatus === "idle" && (
          <button
            type="button"
            onClick={onLoadModel}
            className="group ml-auto inline-flex items-center gap-2 font-display text-sm italic text-cerulean-light transition-colors hover:text-cerulean-light/80"
          >
            <span className="text-base leading-none">↪</span>
            <span className="border-b border-dotted border-cerulean/40 pb-px group-hover:border-cerulean-light">
              load gpt-2 (124M)
            </span>
          </button>
        )}
        {modelStatus === "loading" && (
          <div className="ml-auto h-px w-48 overflow-hidden bg-rule">
            <div
              className="h-full bg-cerulean transition-all duration-300"
              style={{ width: `${loadProgress}%` }}
            />
          </div>
        )}
      </div>
    </header>
  );
}

function StatusInline({
  label,
  status,
  detail,
}: {
  label: string;
  status: "good" | "pending" | "down";
  detail: string;
}) {
  const dotColor =
    status === "good"
      ? "bg-emerald-400"
      : status === "pending"
        ? "bg-amber-400 animate-pulse"
        : "bg-vermillion";
  return (
    <div className="flex items-baseline gap-3">
      <span
        className={`mb-0.5 inline-block h-1.5 w-1.5 rounded-full ${dotColor}`}
      />
      <span className="label-caps text-slate-500">{label}</span>
      <span className="font-mono text-xs text-slate-300">{detail}</span>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────
// Section — Roman numeral + serif title + flex hairline
// ──────────────────────────────────────────────────────────────

function Section({
  number,
  title,
  tryThis,
  children,
}: {
  number: string;
  title: ReactNode;
  /** Optional editor-whisper line under the title — see TryThis component. */
  tryThis?: { hint: string; outcome?: string };
  children: ReactNode;
}) {
  return (
    <section className="mb-20">
      <header
        className={
          tryThis
            ? "mb-2 flex items-baseline gap-5"
            : "mb-8 flex items-baseline gap-5"
        }
      >
        <span className="font-display text-2xl italic text-vermillion">
          {number}.
        </span>
        <h2 className="font-display text-2xl tracking-tight text-graphite">
          {title}
        </h2>
        <span className="rule-flex" />
      </header>
      {tryThis ? (
        <TryThis
          hint={tryThis.hint}
          outcome={tryThis.outcome}
          className="mb-8 ml-9"
        />
      ) : null}
      {children}
    </section>
  );
}

// ──────────────────────────────────────────────────────────────
// Prompt — large editorial textarea
// ──────────────────────────────────────────────────────────────

function PromptInput({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="relative mb-6">
      <span className="label-caps absolute -top-3 left-0 bg-ink px-2 text-slate-500">
        prompt
      </span>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        className="w-full resize-none border border-rule bg-paper px-5 py-4 font-mono text-base leading-relaxed text-graphite placeholder:italic placeholder:text-slate-700 focus:border-vermillion focus:outline-none"
        placeholder="type a prompt to investigate…"
      />
    </div>
  );
}

// ──────────────────────────────────────────────────────────────
// PrimaryAction — editorial text-link button
// ──────────────────────────────────────────────────────────────

function PrimaryAction({
  onClick,
  disabled,
  loading,
  children,
}: {
  onClick: () => void;
  disabled: boolean;
  loading: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="group inline-flex items-center gap-2 font-display text-base italic text-vermillion-light transition-colors hover:text-vermillion-light/90 disabled:cursor-not-allowed disabled:text-slate-700"
    >
      <span className="text-lg leading-none" aria-hidden>
        {loading ? "·" : "↪"}
      </span>
      <span className="border-b border-dotted border-vermillion/40 pb-px group-hover:border-vermillion-light group-disabled:border-slate-800">
        {children}
      </span>
    </button>
  );
}

// ──────────────────────────────────────────────────────────────
// GenerationView — output text + refined telemetry
// ──────────────────────────────────────────────────────────────

function GenerationView({
  text,
  stats,
}: {
  text: string;
  stats: ReturnType<typeof computeTelemetryStats>;
}) {
  return (
    <div className="grid grid-cols-1 gap-10 lg:grid-cols-[1.4fr_1fr]">
      <div>
        <div className="mb-4 flex items-center gap-3">
          <span className="rule-flex" />
          <span className="label-caps text-cerulean">output</span>
          <span className="rule-flex" />
        </div>
        <p className="font-mono text-base leading-relaxed text-graphite">
          {text || (
            <span className="italic font-serif text-slate-700">
              awaiting inference
            </span>
          )}
        </p>
      </div>

      <div>
        <div className="mb-4 flex items-center gap-3">
          <span className="rule-flex" />
          <span className="label-caps text-vermillion">telemetry</span>
          <span className="rule-flex" />
        </div>
        {stats ? (
          <dl className="grid grid-cols-2 gap-x-8 gap-y-2 font-mono text-sm">
            <Stat k="tokens" v={stats.tokens} />
            <Stat k="layers" v={stats.layers} />
            <Stat k="attn tensors" v={stats.attn} />
            <Stat k="hidden tensors" v={stats.hidden} />
            <Stat k="gen steps" v={stats.steps} />
            <Stat k="total memory" v={`${stats.totalMB} MB`} />
          </dl>
        ) : (
          <p className="font-serif italic text-slate-700">
            run inference to capture activations
          </p>
        )}
      </div>
    </div>
  );
}

function Stat({ k, v }: { k: string; v: string | number }) {
  return (
    <>
      <dt className="text-slate-500">{k}</dt>
      <dd className="text-right tabular-nums text-graphite">{v}</dd>
    </>
  );
}

// ──────────────────────────────────────────────────────────────
// Colophon — manuscript-style footer
// ──────────────────────────────────────────────────────────────

function Colophon() {
  return (
    <footer className="mt-32 border-t border-rule pt-8">
      <div className="flex flex-wrap items-baseline justify-between gap-4 font-serif text-xs italic text-slate-500">
        <div>
          NeuroScope <span className="not-italic text-slate-700">·</span> set in
          Fraunces, Newsreader &amp; JetBrains Mono{" "}
          <span className="not-italic text-slate-700">·</span> for the Anthropic
          Fellows portfolio
        </div>
        <div className="font-mono not-italic">
          <a
            href="https://huggingface.co/spaces/lymnal/neuroscope-api"
            className="text-slate-400 hover:text-vermillion-light"
            target="_blank"
            rel="noreferrer"
          >
            api
          </a>
          <span className="mx-2 text-slate-700">·</span>
          <a
            href="https://github.com/lymnal/clearbox_ai"
            className="text-slate-400 hover:text-vermillion-light"
            target="_blank"
            rel="noreferrer"
          >
            source
          </a>
        </div>
      </div>
    </footer>
  );
}
