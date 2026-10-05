//! DGPP (`dgpp-serve`) engine adapter.
//!
//! DGPP exposes the same OpenAI-compatible surface as vLLM (`/health`,
//! `/v1/models`), so model metadata flows through the shared
//! [`super::metadata::ModelMetadata`] machinery. Metrics are the difference:
//! instead of a Prometheus text exposition, DGPP answers `GET /metrics` with
//! a JSON document (`scheduler` / `service` / `prefix_cache` sections) whose
//! lifetime counters and moving averages map onto the dashboard's engine
//! metrics as follows:
//!
//! * `tokens_generated` / `prompt_tokens` — lifetime token counters; the
//!   per-poll deltas drive instantaneous generation/prompt throughput, and
//!   `step_ms` over committed tokens gives true per-token TPOT plus a
//!   per-request TPS (batch-size corrected via the `decode_batch` slot
//!   histogram — see `avg_slots_from_histogram`).
//! * `pool_blocks_in_use` / `pool_blocks_total` — KV-cache pool utilization.
//! * `prefix_cache.hits` / `misses` — prefix-cache hit rate; the paired
//!   `ttft_hit_*` / `ttft_miss_*` averages blend into an overall mean TTFT.
//! * `spec_decode.num_{drafts,draft_tokens,accepted_tokens}_total` — the
//!   same speculative-decoding fields vLLM feeds, so the frontend renders
//!   both engines identically.
//!
//! Throughput and acceptance rates derived from deltas read `None` until the
//! second poll, and the warmup gate (shared `SPARK_WARMUP_SKIP_REQUESTS`
//! knob) blanks histogram-like averages until the engine has served its
//! first request(s).

use super::metadata::ModelMetadata;
use super::{EngineAdapter, EngineMetrics, EngineStatus, EngineType, ModelResolution};
use async_trait::async_trait;
use std::collections::BTreeMap;
use serde::Deserialize;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Default number of requests to skip before reporting averaged/derived
/// metrics. Mirrors the vLLM adapter's default: the first inference on a
/// freshly started engine is dominated by one-time setup costs.
const DEFAULT_WARMUP_SKIP_REQUESTS: u64 = 1;

/// Read the warmup-skip threshold from the environment. Falls back silently
/// to the default on parse failure or when the variable is unset.
fn warmup_skip_from_env() -> u64 {
    std::env::var("SPARK_WARMUP_SKIP_REQUESTS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_WARMUP_SKIP_REQUESTS)
}

/// Token acceptance rate (TAR) as a percentage: `accepted / draft * 100`.
///
/// Returns `None` unless both counts are present, `draft > 0`, and `accepted`
/// is non-negative, so the tile stays blank until at least one token has been
/// drafted and never surfaces a negative rate from a malformed/reset counter.
/// Used for both the lifetime TAR (absolute counters) and the live TAR
/// (per-poll deltas).
fn spec_acceptance_rate(accepted: Option<f64>, draft: Option<f64>) -> Option<f64> {
    match (accepted, draft) {
        (Some(a), Some(d)) if d > 0.0 && a >= 0.0 => Some((a / d) * 100.0),
        _ => None,
    }
}

/// Mean acceptance length: accepted tokens per draft attempt (`accepted / drafts`).
///
/// Returns `None` unless both counts are present and `drafts > 0`.
fn spec_mean_acceptance_length(accepted: Option<f64>, drafts: Option<f64>) -> Option<f64> {
    match (accepted, drafts) {
        (Some(a), Some(n)) if n > 0.0 => Some(a / n),
        _ => None,
    }
}

/// Lifetime weighted-average decode batch size (requests per step) from the
/// engine's `decode_batch.replays_by_slots` histogram: Σ(slots × replays) /
/// Σreplays. `None` while nothing has been decoded or when the histogram is
/// absent (older DGPP), so callers can fall back to a coarser shape.
fn avg_slots_from_histogram(batch: Option<&DgppDecodeBatch>) -> Option<f64> {
    let map = batch?.replays_by_slots.as_ref()?;
    let mut weighted = 0.0f64;
    let mut total = 0.0f64;
    for (slots, count) in map {
        let Ok(slots) = slots.parse::<f64>() else { continue };
        let count = count.max(0.0);
        weighted += slots * count;
        total += count;
    }
    (total > 0.0).then_some(weighted / total)
}

// ---------------------------------------------------------------------------
// GET /metrics response shape
// ---------------------------------------------------------------------------

/// JSON document served at `GET /metrics`. Unknown fields are ignored so a
/// newer DGPP can add sections without breaking the adapter.
#[derive(Deserialize)]
struct DgppMetrics {
    #[serde(default)]
    scheduler: DgppScheduler,
    #[serde(default)]
    service: DgppService,
    #[serde(default)]
    prefix_cache: DgppPrefixCache,
}

#[derive(Deserialize, Default)]
struct DgppScheduler {
    #[serde(default)]
    active: Option<f64>,
    #[serde(default)]
    queued: Option<f64>,
    /// Lifetime generated (decode) tokens.
    #[serde(default)]
    tokens_generated: Option<f64>,
    /// Lifetime prefilled prompts.
    #[serde(default)]
    prompts_prefilled: Option<f64>,
    /// Lifetime prompt (prefill) tokens submitted.
    #[serde(default)]
    prompt_tokens: Option<f64>,
    /// Lifetime decode iteration steps.
    #[serde(default)]
    decode_steps: Option<f64>,
    /// Lifetime decode token rows scheduled (draft attempts included).
    #[serde(default)]
    decode_rows: Option<f64>,
    /// Cumulative prefill time (ms). With `prompts_prefilled`, this is the
    /// fallback mean-TTFT source when the hit/miss breakdown is absent.
    #[serde(default)]
    prefill_ms: Option<f64>,
    /// Cumulative decode step time (ms) — the engine's own moving-average
    /// time base for per-token TPOT and per-request TPS (batch-size
    /// corrected; see `avg_slots` in `compute_metrics`).
    #[serde(default)]
    step_ms: Option<f64>,
    /// Batch-size facts for per-request views; absent on older DGPP.
    #[serde(default)]
    decode_batch: Option<DgppDecodeBatch>,
    /// KV-cache pool blocks: total capacity and currently in use.
    #[serde(default)]
    pool_blocks_total: Option<f64>,
    #[serde(default)]
    pool_blocks_in_use: Option<f64>,
    /// Speculative decoding counters; absent when spec decode is off.
    #[serde(default)]
    spec_decode: Option<DgppSpecDecode>,
}

/// `scheduler.decode_batch`: per-step batch-size facts. Only the lifetime
/// `replays_by_slots` histogram is consumed (keys are slot counts as
/// strings, values are replay counts); everything else is ignored.
#[derive(Deserialize, Default)]
struct DgppDecodeBatch {
    #[serde(default)]
    replays_by_slots: Option<BTreeMap<String, f64>>,
}

#[derive(Deserialize, Default)]
struct DgppSpecDecode {
    /// Verification rounds attempted.
    #[serde(default)]
    num_drafts_total: Option<f64>,
    /// Draft (speculative) token positions attempted.
    #[serde(default)]
    num_draft_tokens_total: Option<f64>,
    /// Draft tokens that passed verification.
    #[serde(default)]
    num_accepted_tokens_total: Option<f64>,
}

#[derive(Deserialize, Default)]
struct DgppService {
    /// Requests that completed serving (the warmup gate reads this).
    #[serde(default)]
    requests_total: Option<f64>,
}

#[derive(Deserialize, Default)]
struct DgppPrefixCache {
    #[serde(default)]
    hits: Option<f64>,
    #[serde(default)]
    misses: Option<f64>,
    /// Average TTFT (ms) over prompts that hit / missed the prefix cache,
    /// with their observation counts. Blended into one mean TTFT.
    #[serde(default)]
    ttft_hit_count: Option<f64>,
    #[serde(default)]
    ttft_hit_ms_avg: Option<f64>,
    #[serde(default)]
    ttft_miss_count: Option<f64>,
    #[serde(default)]
    ttft_miss_ms_avg: Option<f64>,
}

/// Blend the per-outcome TTFT averages into one mean TTFT (ms), weighted by
/// observation count. Handles either half being absent (a cache with only
/// hits or only misses so far) and collapses to `None` when nothing has been
/// observed yet — the UI renders dashes until then.
fn blended_ttft_ms(pc: &DgppPrefixCache) -> Option<f64> {
    match (
        pc.ttft_hit_count,
        pc.ttft_hit_ms_avg,
        pc.ttft_miss_count,
        pc.ttft_miss_ms_avg,
    ) {
        (Some(hn), Some(hm), Some(mn), Some(mm)) => {
            let total = hn + mn;
            (total > 0.0).then(|| (hn * hm + mn * mm) / total)
        }
        (Some(hn), Some(hm), None, None) if hn > 0.0 => Some(hm),
        (None, None, Some(mn), Some(mm)) if mn > 0.0 => Some(mm),
        _ => None,
    }
}

/// Fallback mean TTFT (ms) for engines that do not expose the per-outcome
/// hit/miss breakdown: cumulative prefill time over lifetime prefilled
/// prompts. Collapses to `None` when nothing has been prefilled yet.
fn fallback_ttft_ms(scheduler: &DgppScheduler) -> Option<f64> {
    match (scheduler.prefill_ms, scheduler.prompts_prefilled) {
        (Some(ms), Some(n)) if n > 0.0 => Some(ms / n),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Per-poll derived state
// ---------------------------------------------------------------------------

/// Counter snapshot from the previous poll, for rate computation.
#[derive(Clone, Copy)]
struct DgppCounterSnapshot {
    tokens_generated: f64,
    prompt_tokens: f64,
    at: Instant,
}

/// Everything derived across polls: previous counter readings, running
/// averages, and the previous spec-decode pair for the live acceptance rate.
#[derive(Default)]
struct DgppPollState {
    prev_counters: Option<DgppCounterSnapshot>,
    /// Running average for generation: (sum_of_tps_readings, count_of_readings).
    avg_gen_accum: (f64, u64),
    /// Running average for prompt: (sum_of_tps_readings, count_of_readings).
    avg_prompt_accum: (f64, u64),
    /// Previous (accepted, draft) spec-decode counters for the live TAR.
    prev_spec: Option<(f64, f64)>,
}

/// Warmup gate: blanks averaged/derived metrics until the engine has served
/// `skip` requests. Latches to active — a dashboard attaching to an
/// already-warm engine reports steady state immediately.
struct WarmupGate {
    skip: u64,
    active: bool,
}

impl WarmupGate {
    fn new(skip: u64) -> Self {
        Self { skip, active: false }
    }

    /// Feed the engine's lifetime completed-request count; returns
    /// `(warming_up, just_transitioned)`. A missing count (older DGPP) is
    /// treated as already warm rather than blanking metrics forever.
    fn observe(&mut self, requests_total: Option<u64>) -> (bool, bool) {
        let was = self.active;
        if !self.active {
            self.active = match requests_total {
                Some(r) => r >= self.skip,
                None => true,
            };
        }
        (!self.active, !was && self.active)
    }
}

/// Turn one parsed `/metrics` body plus the previous poll's state into
/// engine metrics. Pure apart from `state`, so tests can drive it without
/// a network.
fn compute_metrics(
    body: &DgppMetrics,
    state: &mut DgppPollState,
    warming_up: bool,
    just_transitioned: bool,
    now: Instant,
) -> EngineMetrics {
    // On the warmup→active transition the previous readings refer to the
    // pre-baseline period; a stale snapshot would yield a misleading first
    // delta, so reset everything delta-derived (mirrors the vLLM adapter).
    if just_transitioned {
        state.prev_counters = None;
        state.avg_gen_accum = (0.0, 0);
        state.avg_prompt_accum = (0.0, 0);
        state.prev_spec = None;
    }

    let scheduler = &body.scheduler;

    // --- Rate metrics from lifetime counter deltas ---
    let gen = scheduler.tokens_generated;
    let prompt = scheduler.prompt_tokens;

    let tokens_per_sec = match (gen, state.prev_counters) {
        (Some(current), Some(prev)) => {
            let elapsed = now.duration_since(prev.at).as_secs_f64();
            (elapsed > 0.0 && current >= prev.tokens_generated)
                .then_some((current - prev.tokens_generated) / elapsed)
        }
        _ => None,
    };
    let prompt_tokens_per_sec = match (prompt, state.prev_counters) {
        (Some(current), Some(prev)) => {
            let elapsed = now.duration_since(prev.at).as_secs_f64();
            (elapsed > 0.0 && current >= prev.prompt_tokens)
                .then_some((current - prev.prompt_tokens) / elapsed)
        }
        _ => None,
    };
    if let (Some(g), Some(p)) = (gen, prompt) {
        state.prev_counters = Some(DgppCounterSnapshot {
            tokens_generated: g,
            prompt_tokens: p,
            at: now,
        });
    }

    // Running averages accumulate only non-zero readings so an idle engine
    // keeps its last average instead of decaying toward zero.
    if let Some(tps) = tokens_per_sec {
        if tps > 0.0 {
            state.avg_gen_accum.0 += tps;
            state.avg_gen_accum.1 += 1;
        }
    }
    if let Some(tps) = prompt_tokens_per_sec {
        if tps > 0.0 {
            state.avg_prompt_accum.0 += tps;
            state.avg_prompt_accum.1 += 1;
        }
    }
    let avg_tokens_per_sec = (state.avg_gen_accum.1 > 0).then(|| state.avg_gen_accum.0 / state.avg_gen_accum.1 as f64);
    let avg_prompt_tokens_per_sec =
        (state.avg_prompt_accum.1 > 0).then(|| state.avg_prompt_accum.0 / state.avg_prompt_accum.1 as f64);

    // Batch-size correction for per-request views. `step_ms`/`decode_steps`
    // is time per decode *step*; with MTP each step commits ~1 sampled token
    // per request plus accepted drafts, and `tokens_generated` counts every
    // committed token across the whole batch — so per-token (and
    // per-request) views divide by the batch size, which the engine exposes
    // as the lifetime `replays_by_slots` histogram. The weighted mean over
    // that histogram matches the engine's own per-request ms/tok (validated
    // against serve-log retire lines); without the histogram fall back to
    // decode_rows/decode_steps (lifetime mean batch rows), which is exact
    // when slots never change but overweights multi-slot steps otherwise.
    let avg_slots = avg_slots_from_histogram(scheduler.decode_batch.as_ref()).or_else(|| {
        match (scheduler.decode_rows, scheduler.decode_steps) {
            (Some(rows), Some(steps)) if steps > 0.0 => Some(rows / steps),
            _ => None,
        }
    });

    // True per-token time: total decode ms over total committed tokens,
    // scaled by the mean batch size (ms/token = step_ms × slots / tokens).
    // Falls back to the engine's raw per-step average when no batch size is
    // derivable, which is exact with speculative decoding off but overstates
    // TPOT under MTP.
    let tpot_ms = match (scheduler.step_ms, scheduler.tokens_generated, avg_slots) {
        (Some(ms), Some(tokens), Some(slots))
            if ms > 0.0 && tokens > 0.0 && slots > 0.0 =>
        {
            Some(ms * slots / tokens)
        }
        (Some(ms), _, None) => scheduler
            .decode_steps
            .filter(|&steps| steps > 0.0 && ms > 0.0)
            .map(|steps| ms / steps),
        _ => None,
    };

    // Per-request decode throughput: committed tokens per second of decode
    // time per batch slot — the same "one request's view" quantity the vLLM
    // adapter derives from its TPOT histogram (validated against the
    // engine's per-request retire lines).
    let per_request_tps = match (scheduler.tokens_generated, scheduler.step_ms, avg_slots) {
        (Some(tokens), Some(ms), Some(slots))
            if tokens > 0.0 && ms > 0.0 && slots > 0.0 =>
        {
            Some(tokens * 1000.0 / (ms * slots))
        }
        _ => None,
    };

    // Per-request prefill throughput: prompt tokens per second of prefill.
    let per_request_prompt_tps = match (scheduler.prompt_tokens, scheduler.prefill_ms) {
        (Some(tokens), Some(ms)) if ms > 0.0 => Some(tokens / (ms / 1000.0)),
        _ => None,
    };

    // --- Pass-through gauges and lifetime counters ---
    let active_requests = scheduler.active.map(|v| v.max(0.0) as u64);
    let queued_requests = scheduler.queued.map(|v| v.max(0.0) as u64);

    let kv_cache_percent = match (scheduler.pool_blocks_in_use, scheduler.pool_blocks_total) {
        (Some(in_use), Some(total)) if total > 0.0 => Some(((in_use / total) * 100.0).clamp(0.0, 100.0)),
        _ => None,
    };

    let total_requests = body.service.requests_total.map(|v| v.max(0.0) as u64);
    let total_prompt_tokens = prompt.map(|v| v.max(0.0) as u64);
    let total_generation_tokens = gen.map(|v| v.max(0.0) as u64);

    let prefix_cache_hit_rate = {
        let hits = body.prefix_cache.hits;
        let misses = body.prefix_cache.misses;
        match (hits, misses) {
            (Some(h), Some(m)) if h >= 0.0 && m >= 0.0 && h + m > 0.0 => Some(h / (h + m) * 100.0),
            _ => None,
        }
    };

    // --- Speculative decoding (same fields/semantics as the vLLM adapter) ---
    let (spec_draft, spec_accepted, spec_drafts) = body
        .scheduler
        .spec_decode
        .as_ref()
        .map(|s| {
            (
                s.num_draft_tokens_total,
                s.num_accepted_tokens_total,
                s.num_drafts_total,
            )
        })
        .unwrap_or((None, None, None));

    // Prometheus counters are non-negative by spec; clamp before the lossy
    // f64->u64 cast so a malformed/negative sample can never wrap.
    let spec_decode_draft_tokens_total = spec_draft.map(|v| v.max(0.0) as u64);
    let spec_decode_accepted_tokens_total = spec_accepted.map(|v| v.max(0.0) as u64);
    let spec_decode_drafts_total = spec_drafts.map(|v| v.max(0.0) as u64);

    let spec_decode_acceptance_rate = spec_acceptance_rate(spec_accepted, spec_draft);
    let spec_decode_mean_acceptance_length = spec_mean_acceptance_length(spec_accepted, spec_drafts);

    // Live (windowed) TAR from per-poll deltas: Δaccepted / Δdraft * 100.
    // The snapshot is discarded whenever a counter is missing or appears to
    // have gone backwards (engine restart), matching the vLLM adapter.
    let spec_decode_acceptance_rate_live = {
        match (spec_accepted, spec_draft) {
            (Some(acc), Some(draft)) => {
                let live = state.prev_spec.as_ref().and_then(|&(prev_acc, prev_draft)| {
                    if acc >= prev_acc && draft >= prev_draft {
                        spec_acceptance_rate(Some(acc - prev_acc), Some(draft - prev_draft))
                    } else {
                        None
                    }
                });
                state.prev_spec = Some((acc, draft));
                live
            }
            _ => {
                state.prev_spec = None;
                None
            }
        }
    };

    // While warming, delta- and average-derived metrics are blanked so the
    // first slow inference does not pollute steady-state numbers; gauges and
    // lifetime counters stay populated so the UI shows live engine state.
    let blank = warming_up;
    EngineMetrics {
        tokens_per_sec: if blank { None } else { tokens_per_sec },
        avg_tokens_per_sec: if blank { None } else { avg_tokens_per_sec },
        ttft_ms: if blank {
            None
        } else {
            blended_ttft_ms(&body.prefix_cache).or_else(|| fallback_ttft_ms(&body.scheduler))
        },
        active_requests,
        queued_requests,
        kv_cache_percent,
        kv_cache_is_estimated: false,
        total_requests,
        prompt_tokens_per_sec: if blank { None } else { prompt_tokens_per_sec },
        avg_prompt_tokens_per_sec: if blank { None } else { avg_prompt_tokens_per_sec },
        per_request_prompt_tps: if blank { None } else { per_request_prompt_tps },
        prefix_cache_hit_rate,
        total_prompt_tokens,
        total_generation_tokens,
        tpot_ms: if blank { None } else { tpot_ms },
        per_request_tps: if blank { None } else { per_request_tps },
        spec_decode_draft_tokens_total,
        spec_decode_accepted_tokens_total,
        spec_decode_drafts_total,
        spec_decode_acceptance_rate,
        spec_decode_acceptance_rate_live: if blank {
            None
        } else {
            spec_decode_acceptance_rate_live
        },
        spec_decode_mean_acceptance_length,
        warming_up,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

pub struct DgppAdapter {
    client: reqwest::Client,
    endpoint: String,
    /// Optional bearer token for an auth-gated deployment. Applied to engine
    /// requests; open endpoints ignore it harmlessly. The HuggingFace request
    /// is never authenticated with it.
    api_key: Option<String>,
    /// `/v1/models` resolution + HuggingFace enrichment, shared with the
    /// vLLM adapter.
    metadata: ModelMetadata,
    /// Per-poll derived state (previous counters, running averages, live TAR).
    poll_state: Mutex<DgppPollState>,
    /// Warmup gate — blanks averaged metrics until the engine has served its
    /// first request(s).
    warming_up: Mutex<WarmupGate>,
}

impl DgppAdapter {
    pub fn new(
        client: reqwest::Client,
        endpoint: String,
        served_model: Option<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client,
            endpoint,
            api_key,
            metadata: ModelMetadata::new(served_model),
            poll_state: Mutex::new(DgppPollState::default()),
            warming_up: Mutex::new(WarmupGate::new(warmup_skip_from_env())),
        }
    }

    /// Attach the bearer token when one is configured. No-op otherwise.
    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => rb.bearer_auth(key),
            None => rb,
        }
    }

    /// Drop every per-poll derived reading — counter snapshot, running
    /// averages, live-TAR pair, warmup gate. The auto-detecting adapter calls
    /// this when the engine flavor at the endpoint changes so rates never
    /// diff across an engine swap.
    pub(super) async fn reset_derived_state(&self) {
        *self.poll_state.lock().await = DgppPollState::default();
        *self.warming_up.lock().await = WarmupGate::new(warmup_skip_from_env());
    }
}

#[async_trait]
impl EngineAdapter for DgppAdapter {
    fn engine_type(&self) -> EngineType {
        EngineType::Dgpp
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    async fn health_check(&self) -> EngineStatus {
        match self
            .auth(
                self.client
                    .get(format!("{}/health", self.endpoint))
                    .timeout(Duration::from_secs(2)),
            )
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => EngineStatus::Running,
            Ok(r) => EngineStatus::Error(format!("HTTP {}", r.status())),
            Err(e) => EngineStatus::Error(e.to_string()),
        }
    }

    async fn get_model_info(&self) -> ModelResolution {
        self.metadata
            .resolve(&self.endpoint, &self.client, self.api_key.as_deref())
            .await
    }

    async fn get_metrics(&self) -> Option<EngineMetrics> {
        let body = self.fetch_metrics_body().await?;
        self.process_metrics_text(&body).await
    }
}

impl DgppAdapter {
    /// Fetch the raw `/metrics` body (a DGPP JSON document).
    pub(super) async fn fetch_metrics_body(&self) -> Option<String> {
        self.auth(
            self.client
                .get(format!("{}/metrics", self.endpoint))
                .timeout(Duration::from_secs(2)),
        )
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .text()
        .await
        .ok()
    }

    /// Parse and process a DGPP JSON `/metrics` body into engine metrics.
    /// Split from the fetch so the auto-detecting adapter can sniff one
    /// fetched body and route it to the right parser. The method name
    /// matches the vLLM adapter's for uniform call sites; the body is JSON.
    pub(super) async fn process_metrics_text(&self, body: &str) -> Option<EngineMetrics> {
        let body = serde_json::from_str::<DgppMetrics>(body).ok()?;

        let requests_total = body.service.requests_total.map(|v| v.max(0.0) as u64);
        let (warming_up, just_transitioned) = self.warming_up.lock().await.observe(requests_total);
        if just_transitioned {
            tracing::info!(
                endpoint = %self.endpoint,
                "warmup complete — steady-state metrics now reported"
            );
        }

        let mut state = self.poll_state.lock().await;
        Some(compute_metrics(
            &body,
            &mut state,
            warming_up,
            just_transitioned,
            Instant::now(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avg_slots_weighted_mean_over_histogram() {
        let mut batch = DgppDecodeBatch::default();
        batch.replays_by_slots = Some(
            [
                ("1".to_string(), 100.0),
                ("4".to_string(), 100.0),
            ]
            .into_iter()
            .collect(),
        );
        // (1×100 + 4×100) / 200 = 2.5.
        assert_eq!(avg_slots_from_histogram(Some(&batch)), Some(2.5));

        batch.replays_by_slots = Some([("1".to_string(), 0.0)].into_iter().collect());
        assert_eq!(avg_slots_from_histogram(Some(&batch)), None, "empty histogram");
        assert_eq!(avg_slots_from_histogram(None), None, "absent decode_batch");
    }

    /// A fixture mirroring the live `dgpp-serve` `/metrics` shape (values
    /// from a real poll). Deserialization must tolerate the full document,
    /// including fields the adapter ignores.
    fn live_fixture() -> DgppMetrics {
        serde_json::from_str(
            r#"{
              "scheduler": {
                "active": 2, "queued": 1, "terminal": 167, "records": 0,
                "record_tokens": 0, "pool_blocks_total": 2240,
                "pool_blocks_in_use": 1835, "prefilling": 0,
                "tokens_generated": 411677, "snapshot_age_ms": 4.384386,
                "prompts_prefilled": 167, "prompt_tokens": 4563370,
                "prompt_tokens_computed": 1671594, "decode_steps": 221417,
                "decode_rows": 231747,
                "decode_batch": {"last_slots": 1, "replays": 221417,
                  "replays_by_slots": {"1": 220617, "2": 800}},
                "spec_decode": {
                  "depth": 1, "num_drafts_total": 231747,
                  "num_draft_tokens_total": 231747,
                  "num_accepted_tokens_total": 179836,
                  "num_draft_tokens_per_pos_total": [231747],
                  "num_accepted_tokens_per_pos_total": [179836]
                },
                "prefill_ms": 6439915.1, "prefill_request_ms": 16135530.4,
                "step_ms": 12946059.9
              },
              "service": {
                "requests_total": 167, "requests_shed": 0,
                "pending_admissions": 0, "tokens_out": 411677
              },
              "prefix_cache": {
                "enabled": true, "slots": 29, "entries": 14,
                "hits": 62, "misses": 105, "tokens_saved": 2891776,
                "ttft_hit_count": 62, "ttft_hit_ms_avg": 27672.931,
                "ttft_miss_count": 105, "ttft_miss_ms_avg": 137882.263
              },
              "prefill": {"requests": [], "prompt_tokens": 0}
            }"#,
        )
        .expect("fixture deserializes")
    }

    #[test]
    fn spec_acceptance_rate_computes_percentage_and_guards_zero() {
        assert_eq!(spec_acceptance_rate(Some(75.0), Some(100.0)), Some(75.0));
        // Zero or missing draft tokens => blank, no divide-by-zero.
        assert_eq!(spec_acceptance_rate(Some(10.0), Some(0.0)), None);
        assert_eq!(spec_acceptance_rate(Some(10.0), None), None);
        assert_eq!(spec_acceptance_rate(None, Some(10.0)), None);
        // A negative numerator (e.g. a counter-reset delta) must not surface a
        // negative rate.
        assert_eq!(spec_acceptance_rate(Some(-5.0), Some(100.0)), None);
    }

    #[test]
    fn spec_mean_acceptance_length_divides_accepted_by_drafts() {
        assert_eq!(
            spec_mean_acceptance_length(Some(300.0), Some(100.0)),
            Some(3.0)
        );
        assert_eq!(spec_mean_acceptance_length(Some(5.0), Some(0.0)), None);
        assert_eq!(spec_mean_acceptance_length(None, Some(2.0)), None);
    }

    #[test]
    fn blended_ttft_weights_both_outcomes() {
        let pc = live_fixture().prefix_cache;
        let expected = (62.0 * 27672.931 + 105.0 * 137882.263) / 167.0;
        let got = blended_ttft_ms(&pc).expect("both halves present");
        assert!((got - expected).abs() < 1e-6, "got {got}, want {expected}");
    }

    #[test]
    fn blended_ttft_handles_one_sided_and_empty_caches() {
        let mut pc = DgppPrefixCache::default();
        assert_eq!(blended_ttft_ms(&pc), None, "nothing observed yet");

        pc.ttft_hit_count = Some(3.0);
        pc.ttft_hit_ms_avg = Some(100.0);
        assert_eq!(blended_ttft_ms(&pc), Some(100.0), "hits only");

        pc.ttft_hit_count = None;
        pc.ttft_hit_ms_avg = None;
        pc.ttft_miss_count = Some(7.0);
        pc.ttft_miss_ms_avg = Some(300.0);
        assert_eq!(blended_ttft_ms(&pc), Some(300.0), "misses only");
    }

    #[test]
    fn fallback_ttft_uses_prefill_totals() {
        let s = live_fixture().scheduler;
        let expected = 6439915.1 / 167.0;
        let got = fallback_ttft_ms(&s).expect("totals present");
        assert!((got - expected).abs() < 1e-6, "got {got}, want {expected}");

        let mut s = DgppScheduler::default();
        assert_eq!(fallback_ttft_ms(&s), None, "nothing prefilled yet");
        s.prefill_ms = Some(1000.0);
        assert_eq!(fallback_ttft_ms(&s), None, "zero prompts guards divide-by-zero");
        s.prompts_prefilled = Some(10.0);
        assert_eq!(fallback_ttft_ms(&s), Some(100.0));
    }

    #[test]
    fn blended_ttft_takes_precedence_over_fallback() {
        let body = live_fixture();
        let blended = blended_ttft_ms(&body.prefix_cache).expect("hit/miss present");
        let fallback = fallback_ttft_ms(&body.scheduler).expect("totals present");
        assert!(blended != fallback, "the hit/miss blend wins when present");
    }

    #[test]
    fn warmup_gate_blank_then_active() {
        let mut gate = WarmupGate::new(1);
        let (warming, transitioned) = gate.observe(Some(0));
        assert!(warming && !transitioned, "no requests served yet");

        let (warming, transitioned) = gate.observe(Some(1));
        assert!(!warming && transitioned, "first request completes warmup");

        let (warming, transitioned) = gate.observe(Some(50));
        assert!(!warming && !transitioned, "stays active, no repeat event");
    }

    #[test]
    fn warmup_gate_treats_missing_counter_as_warm() {
        let mut gate = WarmupGate::new(1);
        let (warming, _) = gate.observe(None);
        assert!(!warming, "older DGPP without the counter must not blank forever");
    }

    #[test]
    fn warmup_gate_latches_when_dashboard_attaches_late() {
        let mut gate = WarmupGate::new(1);
        let (warming, transitioned) = gate.observe(Some(167));
        assert!(!warming, "already-warm engine reports steady state at once");
        assert!(transitioned, "the attach itself is the transition");
    }

    #[test]
    fn first_poll_reports_gauges_but_no_rates() {
        let body = live_fixture();
        let mut state = DgppPollState::default();
        let m = compute_metrics(&body, &mut state, false, false, Instant::now());

        // Pass-through gauges and lifetime counters are populated.
        assert_eq!(m.active_requests, Some(2));
        assert_eq!(m.queued_requests, Some(1));
        assert_eq!(m.total_requests, Some(167));
        assert_eq!(m.total_prompt_tokens, Some(4_563_370));
        assert_eq!(m.total_generation_tokens, Some(411_677));
        assert_eq!(m.spec_decode_draft_tokens_total, Some(231_747));
        assert_eq!(m.spec_decode_accepted_tokens_total, Some(179_836));
        assert_eq!(m.spec_decode_drafts_total, Some(231_747));

        // KV pool: 1835/2240.
        let kv = m.kv_cache_percent.expect("kv percent");
        assert!((kv - 1835.0 / 2240.0 * 100.0).abs() < 1e-9);

        // Prefix hit rate: 62/167.
        let hit = m.prefix_cache_hit_rate.expect("hit rate");
        assert!((hit - 62.0 / 167.0 * 100.0).abs() < 1e-9);

        // TPOT is batch-corrected: step_ms × avg_slots / tokens_generated,
        // with avg_slots the weighted mean of the replays_by_slots histogram
        // (222217 / 221417 here) — ms per committed token, matching the
        // engine's own per-request ms/tok.
        let slots = (220_617.0 + 2.0 * 800.0) / (220_617.0 + 800.0);
        let tpot = m.tpot_ms.expect("tpot");
        assert!((tpot - 12_946_059.9 * slots / 411_677.0).abs() < 1e-6);

        // Per-request TPS is the reciprocal view (committed tokens per decode
        // second per batch slot). Lifetime-derived, so it is available on the
        // first poll — unlike the window-delta rates below.
        let pr = m.per_request_tps.expect("per-request tps");
        assert!((pr - 411_677.0 * 1000.0 / (12_946_059.9 * slots)).abs() < 1e-6);

        // Spec-decode lifetime ratios.
        let tar = m.spec_decode_acceptance_rate.expect("tar");
        assert!((tar - 179836.0 / 231747.0 * 100.0).abs() < 1e-6);
        assert_eq!(m.spec_decode_mean_acceptance_length, Some(179836.0 / 231747.0));

        // Rates need two polls.
        assert_eq!(m.tokens_per_sec, None);
        assert_eq!(m.prompt_tokens_per_sec, None);
        assert_eq!(m.spec_decode_acceptance_rate_live, None);
        // E2E latency has no DGPP source.
        assert_eq!(m.e2e_latency_ms, None);
        assert!(m.ttft_percentiles.is_none());
    }

    #[test]
    fn second_poll_derives_rates_from_counter_deltas() {
        let body = live_fixture();
        let t0 = Instant::now();
        let mut state = DgppPollState::default();
        let _first = compute_metrics(&body, &mut state, false, false, t0);

        // Advance the counters and poll again one second later.
        let mut grown = live_fixture();
        grown.scheduler.tokens_generated = Some(412_677.0);
        grown.scheduler.prompt_tokens = Some(4_573_370.0);
        grown.scheduler.spec_decode.as_mut().unwrap().num_accepted_tokens_total = Some(179_936.0);
        grown.scheduler.spec_decode.as_mut().unwrap().num_draft_tokens_total = Some(231_847.0);

        let t1 = t0 + Duration::from_secs(1);
        let m = compute_metrics(&grown, &mut state, false, false, t1);

        assert!((m.tokens_per_sec.unwrap() - 1000.0).abs() < 1e-6);
        assert!((m.prompt_tokens_per_sec.unwrap() - 10000.0).abs() < 1e-6);
        // Δaccepted/Δdraft over the window: 100/100.
        assert!((m.spec_decode_acceptance_rate_live.unwrap() - 100.0).abs() < 1e-6);
        // Averages accumulated the non-zero readings.
        assert!((m.avg_tokens_per_sec.unwrap() - 1000.0).abs() < 1e-6);
        assert!((m.avg_prompt_tokens_per_sec.unwrap() - 10000.0).abs() < 1e-6);
    }

    #[test]
    fn warmup_blanks_derived_metrics_but_keeps_gauges() {
        let body = live_fixture();
        let mut state = DgppPollState::default();
        let m = compute_metrics(&body, &mut state, true, false, Instant::now());

        assert!(m.warming_up);
        assert_eq!(m.tokens_per_sec, None);
        assert_eq!(m.avg_tokens_per_sec, None);
        assert_eq!(m.tpot_ms, None);
        assert_eq!(m.per_request_tps, None);
        assert_eq!(m.spec_decode_acceptance_rate_live, None);
        // Gauges and lifetime counters stay live during warmup.
        assert_eq!(m.active_requests, Some(2));
        assert_eq!(m.total_generation_tokens, Some(411_677));
        assert_eq!(m.spec_decode_draft_tokens_total, Some(231_747));
    }

    #[test]
    fn transition_resets_delta_state() {
        let body = live_fixture();
        let mut state = DgppPollState::default();
        let _ = compute_metrics(&body, &mut state, true, false, Instant::now());
        assert!(state.prev_counters.is_some());

        // The poll that completes warmup must drop the pre-baseline snapshot
        // instead of diffing across the boundary: no rate on that poll, and a
        // fresh baseline (not the stale pre-baseline one) afterwards.
        let t1 = Instant::now();
        let m = compute_metrics(&body, &mut state, false, true, t1);
        assert!(!m.warming_up);
        assert_eq!(m.tokens_per_sec, None, "no rate across the boundary");
        let fresh = state.prev_counters.expect("fresh baseline recorded");
        assert_eq!(fresh.tokens_generated, 411_677.0);
        assert_eq!(state.avg_gen_accum, (0.0, 0));
        assert!(state.prev_spec.is_some(), "fresh spec pair recorded");

        // The next poll rates normally against the fresh baseline.
        let mut grown = live_fixture();
        grown.scheduler.tokens_generated = Some(412_677.0);
        grown.scheduler.prompt_tokens = Some(4_573_370.0);
        let m = compute_metrics(&grown, &mut state, false, false, t1 + Duration::from_secs(1));
        assert_eq!(m.tokens_per_sec, Some(1000.0));
    }

    #[test]
    fn counter_reset_does_not_surface_negative_rate() {
        let body = live_fixture();
        let t0 = Instant::now();
        let mut state = DgppPollState::default();
        let _ = compute_metrics(&body, &mut state, false, false, t0);

        // Engine restarted: counters went backwards.
        let mut reset = live_fixture();
        reset.scheduler.tokens_generated = Some(100.0);
        let m = compute_metrics(&reset, &mut state, false, false, t0 + Duration::from_secs(1));
        assert_eq!(m.tokens_per_sec, None);
    }
}
