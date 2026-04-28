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

export interface SteeredGenerationResponse {
  prompt: string;
  layer: number;
  alpha: number;
  steered_text: string;
  baseline_text: string;
}

export interface AblationResponse {
  prompt: string;
  layer: number;
  direction_norm_before: number;
  ablated_text: string;
  baseline_text: string;
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

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);

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
): Promise<T> {
  return fetchJson<T>(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
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
