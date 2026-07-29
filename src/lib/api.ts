/**
 * Typed fetch wrapper for the Python FastAPI backend.
 *
 * Each function maps 1:1 to a backend endpoint defined in backend/research.py.
 * All responses are validated for HTTP status before returning typed data.
 */

const API_BASE = (
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
).replace(/\/+$/, "");

// ============================================================================
// Response Interfaces
// ============================================================================

export interface HealthResponse {
  status: string;
  service: string;
}

export interface LoadModelResponse {
  model: string;
  status: string;
  [key: string]: unknown;
}

export interface LogitLensPrediction {
  layer: string;
  top_k: Array<{ token: string; prob: number }>;
}

export interface LogitLensResponse {
  prompt: string;
  tokens: string[];
  predictions: LogitLensPrediction[];
}

export interface AttentionResponse {
  prompt: string;
  tokens: string[];
  layer: number;
  head: number;
  pattern: number[][];
}

export interface GradientNorm {
  token: string;
  norm: number;
  normalized: number;
}

export interface GradientsResponse {
  prompt: string;
  target_token: string;
  tokens: string[];
  gradient_norms: GradientNorm[];
}

export interface SteeringVectorResponse {
  layer: number;
  n_positive: number;
  n_negative: number;
  vector_norm: number;
  vector: number[];
}

export interface ContrastivePairsResponse {
  pairs: Array<{ positive: string; negative: string }>;
  count: number;
}

export interface RefusalPairsResponse {
  pairs: Array<{ harmful: string; harmless: string }>;
  count: number;
}

export interface SteeredGenerationResponse {
  prompt: string;
  layer: number;
  alpha: number;
  steered_text: string;
  baseline_text: string;
}

export interface NextTokenTopK {
  baseline: Array<{ token: string; prob: number }>;
  ablated: Array<{ token: string; prob: number }>;
}

export interface AblationResponse {
  prompt: string;
  layer: number;
  direction_norm_before: number;
  ablated_text: string;
  baseline_text: string;
  next_token_topk?: NextTokenTopK;
}

export interface TechniqueResult {
  name: string;
  paper_url: string;
  layer_used: number;
  refusal_rate_baseline: number;
  refusal_rate_ablated: number;
  delta_refusal_rate: number;
  harmfulness_auc_pre: number;
  harmfulness_auc_post: number;
  delta_auc: number;
  elapsed_seconds: number;
  error: string | null;
  /** |cos(probe weight, ablated direction)|. Optional: absent in older runs. */
  probe_cosine?: number | null;
  /**
   * Uncertainty on the headline numbers (absent in older runs). Refusal-rate
   * CIs are Wilson 95%; AUC CIs are percentile bootstrap 95%. `*_post_p` is the
   * one-sided permutation p that the post-ablation AUC beats chance — i.e.
   * whether the harmfulness signal demonstrably survived rather than just
   * "ΔAUC happened to be small". Tuples are [low, high]; a non-finite bound
   * (NaN from a degenerate bootstrap) arrives as null — the backend sanitizes
   * before serializing.
   */
  refusal_rate_baseline_ci?: [number | null, number | null] | null;
  refusal_rate_ablated_ci?: [number | null, number | null] | null;
  harmfulness_auc_pre_ci?: [number | null, number | null] | null;
  harmfulness_auc_post_ci?: [number | null, number | null] | null;
  harmfulness_auc_post_p?: number | null;
  /**
   * Discriminability = |AUC − 0.5|, range [0, 0.5]. How far the probe is from
   * uninformative, ignoring which direction it points.
   *
   * AUC's no-information point is 0.5, not 0, so raw AUC is the wrong scale for
   * "did the harmfulness signal survive ablation?". A probe at AUC 0.05 reads
   * almost perfectly *backwards* — the information is entirely present, just
   * sign-flipped — yet `harmfulness_auc_post_p` (one-sided, AUC > 0.5) reports
   * it as maximally unsurprising, which reads as "signal gone".
   *
   * `harmfulness_discriminability_post_p` is the TWO-sided permutation p that
   * discriminability exceeds 0, and is the field to prefer. Raw AUC is still
   * worth showing because only its sign carries the direction.
   *
   * Absent on artifacts written before July 29 2026 — treat missing as unknown,
   * never as zero.
   */
  harmfulness_discriminability_pre?: number | null;
  harmfulness_discriminability_post?: number | null;
  harmfulness_discriminability_pre_ci?: [number | null, number | null] | null;
  harmfulness_discriminability_post_ci?: [number | null, number | null] | null;
  harmfulness_discriminability_post_p?: number | null;
  /** Sample sizes behind the intervals. */
  n_refusal_eval?: number | null;
  n_auc_eval?: number | null;
}

export interface BenchResult {
  model_name: string;
  layer: number;
  n_extraction_pairs: number;
  n_eval_prompts: number;
  probe_train_auc: number;
  probe_test_auc: number;
  /** Cross-validated probe AUC — the honest metric when n << d_model. Optional. */
  probe_cv_auc_mean?: number | null;
  probe_cv_auc_std?: number | null;
  /**
   * Run provenance. Present on artifacts produced by
   * backend/scripts/run_bench_local.py; absent on live `/refusal-bench`
   * responses and on any artifact predating July 2026. `device` matters
   * because TransformerLens flags MPS as potentially silently incorrect on
   * torch 2.12 — a reader should be able to see which backend produced a row.
   */
  device?: string | null;
  dtype?: string | null;
  seed?: number | null;
  n_pairs_per_class?: number | null;
  max_new_tokens?: number | null;
  results: TechniqueResult[];
}

export interface PcaTrajectoryPoint {
  token: string;
  token_idx: number;
  layer: number;
  x: number;
  y: number;
  z: number;
}

export interface PcaTrajectoriesResponse {
  prompt: string;
  tokens: string[];
  variance_explained: number[];
  trajectories: PcaTrajectoryPoint[];
}

// ============================================================================
// Internal Helpers
// ============================================================================

/**
 * Default request deadline.
 *
 * Generous on purpose: the deployed backend is a free-tier HuggingFace Space
 * that sleeps after ~30 min idle (~30s to wake) and takes 20–40s more to pull
 * and load a model on a cold start. Anything under ~90s would abort legitimate
 * first requests. Without *any* deadline a slept Space leaves the UI spinning
 * with no error and no recovery, which is the worse failure.
 */
const DEFAULT_TIMEOUT_MS = 120_000;

/**
 * `/refusal-bench` runs real generation over every eval prompt for every
 * requested technique — minutes locally, and longer than the Space's proxy
 * allows (see docs/DEPLOYMENT.md §1e). It gets its own ceiling rather than the
 * default so a legitimate long local run is not cut off at two minutes.
 */
const BENCH_TIMEOUT_MS = 1_800_000;

async function fetchJson<T>(
  url: string,
  init?: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  // AbortSignal.timeout() is not in Safari < 16, and this app already targets
  // browsers new enough for WebGPU — but an explicit controller also lets the
  // abort reason be a readable message instead of a bare TimeoutError.
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new Error(`Request timed out after ${timeoutMs}ms`)),
    timeoutMs,
  );

  let response: Response;
  try {
    response = await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    if (controller.signal.aborted) {
      throw new Error(
        `Request to ${url} timed out after ${Math.round(timeoutMs / 1000)}s. ` +
          `If this is the deployed backend, the Space may be asleep or the ` +
          `request may be too heavy for the free CPU tier.`,
      );
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    const body = await response.text().catch(() => "No response body");
    throw new Error(
      `API request failed: ${response.status} ${response.statusText} — ${body}`,
    );
  }

  return response.json() as Promise<T>;
}

function postJson<T>(
  endpoint: string,
  body: Record<string, unknown>,
  timeoutMs?: number,
): Promise<T> {
  return fetchJson<T>(
    `${API_BASE}${endpoint}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    timeoutMs,
  );
}

// ============================================================================
// Endpoint Functions
// ============================================================================

/** GET / — Backend health check. */
export function getHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>(`${API_BASE}/`);
}

/** POST /load — Load a model by name. */
export function loadModel(modelName: string): Promise<LoadModelResponse> {
  return postJson<LoadModelResponse>("/load", { model_name: modelName });
}

/** POST /logit-lens — Run logit lens analysis on a prompt. */
export function getLogitLens(prompt: string): Promise<LogitLensResponse> {
  return postJson<LogitLensResponse>("/logit-lens", { prompt });
}

/** POST /attention — Get attention pattern for a specific layer and head. */
export function getAttention(
  prompt: string,
  layer: number,
  head: number,
): Promise<AttentionResponse> {
  return postJson<AttentionResponse>("/attention", { prompt, layer, head });
}

/** POST /gradients — Compute gradient norms w.r.t. a target token. */
export function getGradients(
  prompt: string,
  targetToken: string,
): Promise<GradientsResponse> {
  return postJson<GradientsResponse>("/gradients", {
    prompt,
    target_token: targetToken,
  });
}

/** POST /steering-vector — Compute a steering vector from contrastive prompts. */
export function getSteeringVector(
  positivePrompts: string[],
  negativePrompts: string[],
  layer: number,
): Promise<SteeringVectorResponse> {
  return postJson<SteeringVectorResponse>("/steering-vector", {
    positive_prompts: positivePrompts,
    negative_prompts: negativePrompts,
    layer,
  });
}

/** GET /contrastive-pairs — Fetch built-in contrastive prompt pairs. */
export function getContrastivePairs(): Promise<ContrastivePairsResponse> {
  return fetchJson<ContrastivePairsResponse>(`${API_BASE}/contrastive-pairs`);
}

/** GET /refusal-pairs — Fetch curated refusal-direction contrastive pairs. */
export function getRefusalPairs(): Promise<RefusalPairsResponse> {
  return fetchJson<RefusalPairsResponse>(`${API_BASE}/refusal-pairs`);
}

/** POST /generate-steered — Generate text with a steering vector applied. */
export function generateSteered(
  prompt: string,
  steeringVector: number[],
  alpha: number,
  layer: number,
  maxNewTokens: number = 30,
): Promise<SteeredGenerationResponse> {
  return postJson<SteeredGenerationResponse>("/generate-steered", {
    prompt,
    steering_vector: steeringVector,
    alpha,
    layer,
    max_new_tokens: maxNewTokens,
  });
}

/** POST /pca-trajectories — Compute PCA trajectories across layers. */
export function getPcaTrajectories(
  prompt: string,
): Promise<PcaTrajectoriesResponse> {
  return postJson<PcaTrajectoriesResponse>("/pca-trajectories", { prompt });
}

/** POST /ablate-direction — Generate with a direction projected out. */
export function ablateDirection(
  prompt: string,
  direction: number[],
  layer: number,
  maxNewTokens: number = 30,
): Promise<AblationResponse> {
  return postJson<AblationResponse>("/ablate-direction", {
    prompt,
    direction,
    layer,
    max_new_tokens: maxNewTokens,
  });
}

/**
 * POST /refusal-bench — Head-to-head Refusal Bench.
 *
 * Trains a shared harmfulness probe on the extraction split, then loops over
 * every requested technique and scores it on the held-out eval split.
 * The two-axis result (Δ refusal-rate vs Δ AUC) is the headline novelty.
 */
export function runBench(req: {
  technique_names: string[];
  layer: number;
  harmful_prompts: string[];
  harmless_prompts: string[];
  test_fraction?: number;
  max_new_tokens?: number;
  temperature?: number;
  seed?: number;
}): Promise<BenchResult> {
  return postJson<BenchResult>("/refusal-bench", { ...req }, BENCH_TIMEOUT_MS);
}
