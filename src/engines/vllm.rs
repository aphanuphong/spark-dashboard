use super::histogram::{fraction_le, percentile};
use super::metadata::ModelMetadata;
use super::prometheus::parse_prometheus_text;
use super::warmup::WarmupTracker;
use super::{
    EngineAdapter, EngineMetrics, EngineStatus, EngineType, HistogramBucket, LatencyPercentiles,
    ModelResolution, E2E_SLO_MS, ITL_SLO_MS, TPOT_SLO_MS, TTFT_SLO_MS,
};
use async_trait::async_trait;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Default number of requests to skip on engine startup before baselining.
/// vLLM's first inference is dominated by CUDA kernel JIT and KV cache
/// allocation; excluding one request consistently removes the outlier.
const DEFAULT_WARMUP_SKIP_REQUESTS: u64 = 1;

/// Read the warmup-skip threshold from the environment. Falls back silently
/// to the default on parse failure or when the variable is unset.
fn warmup_skip_from_env() -> u64 {
    std::env::var("SPARK_WARMUP_SKIP_REQUESTS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_WARMUP_SKIP_REQUESTS)
}

pub struct VllmAdapter {
    client: reqwest::Client,
    endpoint: String,
    /// Optional bearer token for auth-gated deployments (vLLM `--api-key`).
    /// Applied to engine requests; open endpoints (`/health`, `/metrics`)
    /// ignore it harmlessly. The HuggingFace request is never authenticated
    /// with it.
    api_key: Option<String>,
    /// `/v1/models` resolution + HuggingFace enrichment. Holds the
    /// launch-command-line model id as the fallback hint when `/v1/models`
    /// returns a bare slug without the HF-style `Provider/` prefix.
    metadata: ModelMetadata,
    /// Previous generation_tokens_total counter reading for rate computation.
    prev_gen_tokens: Mutex<Option<(f64, Instant)>>,
    /// Previous prompt_tokens_total counter reading for rate computation.
    prev_prompt_tokens: Mutex<Option<(f64, Instant)>>,
    /// Previous (accepted, draft) spec-decode counter readings, used to compute
    /// the live (windowed) token acceptance rate from per-poll deltas. No
    /// timestamp is stored because the live TAR is a unit-free ratio of deltas.
    prev_spec_decode: Mutex<Option<(f64, f64)>>,
    /// Running average for generation: (sum_of_tps_readings, count_of_readings)
    avg_accum: Mutex<(f64, u64)>,
    /// Running average for prompt: (sum_of_tps_readings, count_of_readings)
    avg_prompt_accum: Mutex<(f64, u64)>,
    /// Warmup baseline tracker — drops the first `SPARK_WARMUP_SKIP_REQUESTS`
    /// observations from histogram-derived metrics so the slow first inference
    /// does not skew steady-state percentiles and averages.
    warmup: Mutex<WarmupTracker>,
}

impl VllmAdapter {
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
            prev_gen_tokens: Mutex::new(None),
            prev_prompt_tokens: Mutex::new(None),
            prev_spec_decode: Mutex::new(None),
            avg_accum: Mutex::new((0.0, 0)),
            avg_prompt_accum: Mutex::new((0.0, 0)),
            warmup: Mutex::new(WarmupTracker::new(warmup_skip_from_env())),
        }
    }

    /// Attach the bearer token when one is configured. No-op otherwise, so
    /// open `/health` and `/metrics` endpoints are unaffected.
    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => rb.bearer_auth(key),
            None => rb,
        }
    }

    /// Drop every per-poll derived reading — counter snapshots, running
    /// averages, live-TAR pair, warmup baseline. The auto-detecting adapter
    /// calls this when the engine flavor at the endpoint changes so rates
    /// never diff across an engine swap.
    pub(super) async fn reset_derived_state(&self) {
        *self.prev_gen_tokens.lock().await = None;
        *self.prev_prompt_tokens.lock().await = None;
        *self.prev_spec_decode.lock().await = None;
        *self.avg_accum.lock().await = (0.0, 0);
        *self.avg_prompt_accum.lock().await = (0.0, 0);
        *self.warmup.lock().await = WarmupTracker::new(warmup_skip_from_env());
    }

}

#[async_trait]
impl EngineAdapter for VllmAdapter {
    fn engine_type(&self) -> EngineType {
        EngineType::Vllm
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

impl VllmAdapter {
    /// Fetch the raw Prometheus `/metrics` body.
    pub(super) async fn fetch_metrics_body(&self) -> Option<String> {
        self.auth(
            self.client
                .get(format!("{}/metrics", self.endpoint))
                .timeout(Duration::from_secs(2)),
        )
        .send()
        .await
        .ok()?
        .text()
        .await
        .ok()
    }

    /// Parse and process a Prometheus `/metrics` body into engine metrics.
    /// Split from the fetch so the auto-detecting adapter can sniff one
    /// fetched body and route it to the right parser.
    pub(super) async fn process_metrics_text(&self, body: &str) -> Option<EngineMetrics> {
        let raw = parse_prometheus_text(body)?;

        // Run the parsed metrics through the warmup tracker. While warming, the
        // tracker hands back gauges and counters as-is so pass-through fields
        // (active/queued/kv_cache) stay populated; histogram-derived fields
        // are then forced to None below. After baselining, `adjusted` contains
        // counter and histogram deltas — feeding them into the existing
        // `percentile`/`fraction_le` helpers yields warmup-free metrics.
        let warmup_out = {
            let mut tracker = self.warmup.lock().await;
            tracker.observe(&raw)
        };

        // On baseline transition, the per-poll rate state captured during
        // warmup refers to absolute counter values. Post-transition the
        // tracker hands back deltas, so a stale `prev_*` reading would yield
        // a hugely negative rate on the next tick. Reset everything that
        // depends on the previous reading before computing rates below.
        if warmup_out.just_transitioned {
            *self.prev_gen_tokens.lock().await = None;
            *self.prev_prompt_tokens.lock().await = None;
            *self.prev_spec_decode.lock().await = None;
            *self.avg_accum.lock().await = (0.0, 0);
            *self.avg_prompt_accum.lock().await = (0.0, 0);
            tracing::info!(
                endpoint = %self.endpoint,
                "warmup complete — baseline captured, steady-state metrics now reported"
            );
        }

        let parsed = &warmup_out.adjusted;
        let warming_up = warmup_out.warming_up;

        let active_requests = parsed
            .gauges
            .get("vllm_num_requests_running")
            .map(|v| *v as u64);
        let queued_requests = parsed
            .gauges
            .get("vllm_num_requests_waiting")
            .map(|v| *v as u64);
        // v1 uses vllm_kv_cache_usage_perc, v0.6 uses vllm_gpu_cache_usage_perc
        let kv_cache_percent = parsed
            .gauges
            .get("vllm_kv_cache_usage_perc")
            .or_else(|| parsed.gauges.get("vllm_gpu_cache_usage_perc"))
            .map(|v| v * 100.0);

        // TTFT from histogram sum/count (average)
        let ttft_count = parsed
            .counters
            .get("vllm_time_to_first_token_seconds_count");
        let ttft_ms = {
            let sum = parsed.counters.get("vllm_time_to_first_token_seconds_sum");
            match (sum, ttft_count) {
                (Some(&s), Some(&c)) if c > 0.0 => Some((s / c) * 1000.0),
                _ => None,
            }
        };

        // total_requests is a pass-through display field — show the engine's
        // absolute lifetime request count, not the post-baseline delta. Read
        // from `raw` so the value stays continuous across the warmup→active
        // transition rather than snapping back to zero.
        let total_requests = raw
            .counters
            .get("vllm_time_to_first_token_seconds_count")
            .map(|&c| c as u64);

        // Per-request avg TPS from time_per_output_token histogram: 1 / avg_TPOT
        // v1: vllm_request_time_per_output_token_seconds, v0.6: vllm_time_per_output_token_seconds
        let per_request_tps = {
            let sum = parsed
                .counters
                .get("vllm_request_time_per_output_token_seconds_sum")
                .or_else(|| {
                    parsed
                        .counters
                        .get("vllm_time_per_output_token_seconds_sum")
                });
            let count = parsed
                .counters
                .get("vllm_request_time_per_output_token_seconds_count")
                .or_else(|| {
                    parsed
                        .counters
                        .get("vllm_time_per_output_token_seconds_count")
                });
            match (sum, count) {
                (Some(&s), Some(&c)) if c > 0.0 && s > 0.0 => Some(c / s),
                _ => None,
            }
        };

        // TPS from generation_tokens_total counter (rate = delta / elapsed)
        let current_gen = parsed.counters.get("vllm_generation_tokens_total").copied();
        let now = Instant::now();

        let tokens_per_sec = {
            let mut prev_lock = self.prev_gen_tokens.lock().await;
            let tps = match (current_gen, prev_lock.as_ref()) {
                (Some(current), Some(&(prev_val, prev_time))) => {
                    let elapsed = now.duration_since(prev_time).as_secs_f64();
                    if elapsed > 0.0 {
                        Some((current - prev_val) / elapsed)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(val) = current_gen {
                *prev_lock = Some((val, now));
            }
            tps
        };

        // Prompt tokens/sec from prompt_tokens_total counter (rate = delta / elapsed)
        let current_prompt = parsed.counters.get("vllm_prompt_tokens_total").copied();
        let prompt_tokens_per_sec = {
            let mut prev_lock = self.prev_prompt_tokens.lock().await;
            let tps = match (current_prompt, prev_lock.as_ref()) {
                (Some(current), Some(&(prev_val, prev_time))) => {
                    let elapsed = now.duration_since(prev_time).as_secs_f64();
                    if elapsed > 0.0 {
                        Some((current - prev_val) / elapsed)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(val) = current_prompt {
                *prev_lock = Some((val, now));
            }
            tps
        };

        // Avg TPS = sum of non-zero TPS readings / count of readings.
        // Only accumulates when there's actual throughput. Stays stable when idle.
        let avg_tokens_per_sec = {
            let mut accum = self.avg_accum.lock().await;
            if let Some(tps) = tokens_per_sec {
                if tps > 0.0 {
                    accum.0 += tps;
                    accum.1 += 1;
                }
            }
            if accum.1 > 0 {
                Some(accum.0 / accum.1 as f64)
            } else {
                None
            }
        };

        // Avg prompt TPS (same pattern as generation avg)
        let avg_prompt_tokens_per_sec = {
            let mut accum = self.avg_prompt_accum.lock().await;
            if let Some(tps) = prompt_tokens_per_sec {
                if tps > 0.0 {
                    accum.0 += tps;
                    accum.1 += 1;
                }
            }
            if accum.1 > 0 {
                Some(accum.0 / accum.1 as f64)
            } else {
                None
            }
        };

        // Per-request prompt TPS: prompt_tokens_total / ttft_total_seconds
        // Approximates average prefill throughput per request
        let per_request_prompt_tps = {
            let prompt_total = parsed.counters.get("vllm_prompt_tokens_total");
            let ttft_sum = parsed.counters.get("vllm_time_to_first_token_seconds_sum");
            match (prompt_total, ttft_sum) {
                (Some(&p), Some(&t)) if t > 0.0 => Some(p / t),
                _ => None,
            }
        };

        // --- New metrics ---

        // End-to-end request latency (avg from histogram)
        let e2e_latency_ms = {
            let sum = parsed.counters.get("vllm_e2e_request_latency_seconds_sum");
            let count = parsed
                .counters
                .get("vllm_e2e_request_latency_seconds_count");
            match (sum, count) {
                (Some(&s), Some(&c)) if c > 0.0 => Some((s / c) * 1000.0),
                _ => None,
            }
        };

        // Swapped requests (memory pressure indicator)
        let swapped_requests = parsed
            .gauges
            .get("vllm_num_requests_swapped")
            .map(|v| *v as u64);

        // Prefix cache hit rate as percentage, computed from the two counters
        // vLLM exposes (vllm:prefix_cache_hits / vllm:prefix_cache_queries).
        // Guard against queries == 0 so the tile stays blank until the engine
        // has served at least one prompt.
        let prefix_cache_hit_rate = {
            let hits = parsed.counters.get("vllm_prefix_cache_hits_total");
            let queries = parsed.counters.get("vllm_prefix_cache_queries_total");
            match (hits, queries) {
                (Some(&h), Some(&q)) if q > 0.0 => Some((h / q) * 100.0),
                _ => None,
            }
        };

        // Cumulative prefix-cache queries — pass-through lifetime counter, the
        // volume the hit rate is derived from. Mirrors total_*_tokens: shown
        // raw and ungated by warmup so it stays continuous.
        let prefix_cache_queries_total = parsed
            .counters
            .get("vllm_prefix_cache_queries_total")
            .map(|&q| q as u64);

        // Average queue wait time (from histogram)
        let queue_time_ms = {
            let sum = parsed.counters.get("vllm_request_queue_time_seconds_sum");
            let count = parsed.counters.get("vllm_request_queue_time_seconds_count");
            match (sum, count) {
                (Some(&s), Some(&c)) if c > 0.0 => Some((s / c) * 1000.0),
                _ => None,
            }
        };

        // Total preemptions — pass-through display field, read absolute value
        // from `raw` so the lifetime count stays continuous across baselining.
        let preemptions_total = raw
            .counters
            .get("vllm_num_preemptions_total")
            .map(|v| *v as u64);

        // Speculative decoding. vLLM only emits these counters when the served
        // model has speculative decoding configured, so their presence is the
        // signal the frontend uses to show the section at all. The values are
        // read from `raw` (absolute lifetime) so the cumulative token counters
        // count up continuously and the lifetime acceptance ratios are stable.
        // vLLM's prometheus_client appends `_total` to counter names, so try the
        // `_total` key first and fall back to the bare logical name.
        let spec_counter = |name: &str| -> Option<f64> {
            raw.counters
                .get(&format!("vllm_spec_decode_{name}_total"))
                .or_else(|| raw.counters.get(&format!("vllm_spec_decode_{name}")))
                .copied()
        };
        let spec_draft = spec_counter("num_draft_tokens");
        let spec_accepted = spec_counter("num_accepted_tokens");
        let spec_drafts = spec_counter("num_drafts");

        // Prometheus counters are non-negative by spec; clamp before the lossy
        // f64->u64 cast so a malformed/negative sample can never wrap to a huge
        // value (same boundary as `preemptions_total` / `total_requests`).
        let spec_decode_draft_tokens_total = spec_draft.map(|v| v.max(0.0) as u64);
        let spec_decode_accepted_tokens_total = spec_accepted.map(|v| v.max(0.0) as u64);
        let spec_decode_drafts_total = spec_drafts.map(|v| v.max(0.0) as u64);

        // Lifetime token acceptance rate (TAR) = accepted / draft * 100.
        let spec_decode_acceptance_rate = spec_acceptance_rate(spec_accepted, spec_draft);
        // Mean acceptance length = accepted tokens / draft attempts.
        let spec_decode_mean_acceptance_length =
            spec_mean_acceptance_length(spec_accepted, spec_drafts);
        // Live (windowed) TAR from per-poll deltas: Δaccepted / Δdraft * 100.
        // The snapshot is discarded (`prev = None`) whenever a counter is
        // missing this poll or appears to have gone backwards (engine restart
        // resets its counters). Both cases would otherwise diff against a stale
        // snapshot and silently inflate or negate the live rate.
        let spec_decode_acceptance_rate_live = {
            let mut prev_lock = self.prev_spec_decode.lock().await;
            match (spec_accepted, spec_draft) {
                (Some(acc), Some(draft)) => {
                    let live = prev_lock.as_ref().and_then(|&(prev_acc, prev_draft)| {
                        if acc >= prev_acc && draft >= prev_draft {
                            spec_acceptance_rate(Some(acc - prev_acc), Some(draft - prev_draft))
                        } else {
                            None
                        }
                    });
                    *prev_lock = Some((acc, draft));
                    live
                }
                _ => {
                    *prev_lock = None;
                    None
                }
            }
        };

        // Average batch size (tokens per iteration step)
        let avg_batch_size = {
            let sum = parsed.counters.get("vllm_iteration_tokens_total_sum");
            let count = parsed.counters.get("vllm_iteration_tokens_total_count");
            match (sum, count) {
                (Some(&s), Some(&c)) if c > 0.0 => Some(s / c),
                _ => None,
            }
        };

        // Average inter-token latency during decode (from histogram).
        // Guard against count == 0 so the tile stays blank until the engine
        // has streamed at least one inter-token gap.
        let inter_token_latency_ms = {
            let sum = parsed.counters.get("vllm_inter_token_latency_seconds_sum");
            let count = parsed
                .counters
                .get("vllm_inter_token_latency_seconds_count");
            match (sum, count) {
                (Some(&s), Some(&c)) if c > 0.0 => Some((s / c) * 1000.0),
                _ => None,
            }
        };

        // Average time per output token during decode (from histogram
        // sum/count). vLLM v1 names it `request_time_per_output_token`;
        // v0.6 used `time_per_output_token`. Same shape as ttft_ms.
        let tpot_ms = {
            let sum = parsed
                .counters
                .get("vllm_request_time_per_output_token_seconds_sum")
                .or_else(|| {
                    parsed
                        .counters
                        .get("vllm_time_per_output_token_seconds_sum")
                });
            let count = parsed
                .counters
                .get("vllm_request_time_per_output_token_seconds_count")
                .or_else(|| {
                    parsed
                        .counters
                        .get("vllm_time_per_output_token_seconds_count")
                });
            match (sum, count) {
                (Some(&s), Some(&c)) if c > 0.0 => Some((s / c) * 1000.0),
                _ => None,
            }
        };

        // Tail latency percentiles. vLLM exposes `_bucket{le="..."}` lines for
        // each request-level histogram. We linearly interpolate p50/p95/p99 in
        // milliseconds (engine emits seconds). Returns `None` for the whole
        // struct when no buckets exist or the engine has not observed any
        // requests yet — the UI then renders dashes.
        let percentiles_ms = |metric: &str| -> Option<LatencyPercentiles> {
            let buckets = parsed.histograms.get(metric)?;
            let to_ms = |q: f64| percentile(buckets, q).map(|s| s * 1000.0);
            let p = LatencyPercentiles {
                p50_ms: to_ms(0.50),
                p95_ms: to_ms(0.95),
                p99_ms: to_ms(0.99),
            };
            // If every quantile is None (e.g. only +Inf bucket present),
            // collapse to None so the JSON payload stays compact.
            if p.p50_ms.is_none() && p.p95_ms.is_none() && p.p99_ms.is_none() {
                None
            } else {
                Some(p)
            }
        };
        let ttft_percentiles = percentiles_ms("vllm_time_to_first_token_seconds");
        let itl_percentiles = percentiles_ms("vllm_inter_token_latency_seconds");
        let e2e_percentiles = percentiles_ms("vllm_e2e_request_latency_seconds");

        // Goodput: % of histogram observations meeting the SLO. Thresholds
        // come in milliseconds; the histograms are in seconds, so divide by
        // 1000 before passing to fraction_le.
        let goodput_pct = |metric: &str, slo_ms: f64| -> Option<f64> {
            let buckets = parsed.histograms.get(metric)?;
            fraction_le(buckets, slo_ms / 1000.0).map(|f| f * 100.0)
        };
        let ttft_goodput_pct = goodput_pct("vllm_time_to_first_token_seconds", TTFT_SLO_MS);
        let itl_goodput_pct = goodput_pct("vllm_inter_token_latency_seconds", ITL_SLO_MS);
        let e2e_goodput_pct = goodput_pct("vllm_e2e_request_latency_seconds", E2E_SLO_MS);

        // Raw histogram buckets shipped to the frontend so it can recompute
        // goodput at user-customized SLO thresholds without a roundtrip.
        // `+Inf` is replaced with `f64::MAX` because serde_json cannot encode
        // non-finite floats — the frontend port of `fraction_le` treats that
        // sentinel as the overflow bucket.
        let buckets_for = |metric: &str| -> Option<Vec<HistogramBucket>> {
            let raw = parsed.histograms.get(metric)?;
            if raw.is_empty() {
                return None;
            }
            Some(
                raw.iter()
                    .map(|&(le, cum)| HistogramBucket {
                        le_seconds: if le.is_finite() { le } else { f64::MAX },
                        cumulative_count: cum,
                    })
                    .collect(),
            )
        };
        let ttft_buckets = buckets_for("vllm_time_to_first_token_seconds");
        let itl_buckets = buckets_for("vllm_inter_token_latency_seconds");
        let e2e_buckets = buckets_for("vllm_e2e_request_latency_seconds");

        // TPOT has two possible histogram keys depending on vLLM version.
        // Resolve whichever is present, then reuse the closures above.
        let tpot_hist = if parsed
            .histograms
            .contains_key("vllm_request_time_per_output_token_seconds")
        {
            Some("vllm_request_time_per_output_token_seconds")
        } else if parsed
            .histograms
            .contains_key("vllm_time_per_output_token_seconds")
        {
            Some("vllm_time_per_output_token_seconds")
        } else {
            None
        };
        let tpot_percentiles = tpot_hist.and_then(percentiles_ms);
        let tpot_goodput_pct = tpot_hist.and_then(|m| goodput_pct(m, TPOT_SLO_MS));
        let tpot_buckets = tpot_hist.and_then(buckets_for);

        // While warming, histogram-derived metrics still compute from raw
        // pass-through counters/buckets (the tracker doesn't yet have a
        // baseline), so they would carry the slow first observation. Force
        // those fields to None until the tracker transitions to Active. Pass-
        // through fields (gauges + total/preemptions/prefix-cache-rate) stay
        // populated so the UI keeps showing live engine state during warmup.
        let blank = warming_up;
        Some(EngineMetrics {
            tokens_per_sec: if blank { None } else { tokens_per_sec },
            avg_tokens_per_sec: if blank { None } else { avg_tokens_per_sec },
            per_request_tps: if blank { None } else { per_request_tps },
            ttft_ms: if blank { None } else { ttft_ms },
            active_requests,
            queued_requests,
            kv_cache_percent,
            kv_cache_is_estimated: false,
            total_requests,
            e2e_latency_ms: if blank { None } else { e2e_latency_ms },
            prompt_tokens_per_sec: if blank { None } else { prompt_tokens_per_sec },
            avg_prompt_tokens_per_sec: if blank {
                None
            } else {
                avg_prompt_tokens_per_sec
            },
            per_request_prompt_tps: if blank { None } else { per_request_prompt_tps },
            swapped_requests,
            prefix_cache_hit_rate,
            queue_time_ms: if blank { None } else { queue_time_ms },
            inter_token_latency_ms: if blank { None } else { inter_token_latency_ms },
            preemptions_total,
            total_prompt_tokens: current_prompt.map(|v| v as u64),
            total_generation_tokens: current_gen.map(|v| v as u64),
            prefix_cache_queries_total,
            avg_batch_size: if blank { None } else { avg_batch_size },
            ttft_percentiles: if blank { None } else { ttft_percentiles },
            itl_percentiles: if blank { None } else { itl_percentiles },
            e2e_percentiles: if blank { None } else { e2e_percentiles },
            ttft_goodput_pct: if blank { None } else { ttft_goodput_pct },
            itl_goodput_pct: if blank { None } else { itl_goodput_pct },
            e2e_goodput_pct: if blank { None } else { e2e_goodput_pct },
            ttft_buckets: if blank { None } else { ttft_buckets },
            itl_buckets: if blank { None } else { itl_buckets },
            e2e_buckets: if blank { None } else { e2e_buckets },
            tpot_ms: if blank { None } else { tpot_ms },
            tpot_percentiles: if blank { None } else { tpot_percentiles },
            tpot_goodput_pct: if blank { None } else { tpot_goodput_pct },
            tpot_buckets: if blank { None } else { tpot_buckets },
            // Cumulative counters and lifetime ratios are pass-through (ungated
            // by warmup) so they count up / stay stable continuously. Live TAR
            // is a delta-derived rate, so it is blanked during warmup like the
            // other per-poll rates.
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
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check: percentiles flow from the parser through the adapter.
    /// We don't spin up an HTTP mock here — `parse_prometheus_text` is the
    /// boundary we care about, so we assert the percentile pipeline against
    /// a representative `/metrics` body. p50 < p95 < p99 must hold.
    #[test]
    fn ttft_percentiles_roundtrip_from_metrics_body() {
        let body = "\
# HELP vllm:time_to_first_token_seconds TTFT histogram.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le=\"0.05\"} 50
vllm:time_to_first_token_seconds_bucket{le=\"0.1\"} 80
vllm:time_to_first_token_seconds_bucket{le=\"0.5\"} 95
vllm:time_to_first_token_seconds_bucket{le=\"1.0\"} 99
vllm:time_to_first_token_seconds_bucket{le=\"+Inf\"} 100
vllm:time_to_first_token_seconds_sum 12.0
vllm:time_to_first_token_seconds_count 100.0
";
        let parsed = parse_prometheus_text(body).expect("parse");
        let buckets = parsed
            .histograms
            .get("vllm_time_to_first_token_seconds")
            .expect("histogram");
        let p50 = percentile(buckets, 0.5).expect("p50") * 1000.0;
        let p95 = percentile(buckets, 0.95).expect("p95") * 1000.0;
        let p99 = percentile(buckets, 0.99).expect("p99") * 1000.0;
        assert!(p50 < p95, "p50 {p50} < p95 {p95}");
        assert!(p95 < p99, "p95 {p95} < p99 {p99}");
        // p50 lands at the 0.05 boundary (cumulative count exactly 50).
        // p99 lands inside the (0.5, 1.0] bucket.
        assert!((40.0..=60.0).contains(&p50), "p50 {p50} near 50ms");
        assert!(p99 > 500.0 && p99 <= 1000.0, "p99 {p99} in (500, 1000]");
    }

    /// TPOT pipeline: vLLM v1 emits `request_time_per_output_token_seconds`.
    /// The histogram must land under the normalized key (so `tpot_hist`
    /// resolves it), percentiles must order p50<p95<p99, and the sum/count
    /// counters must yield the same average TPOT the adapter computes.
    #[test]
    fn tpot_histogram_roundtrip_from_metrics_body() {
        let body = "\
# HELP vllm:request_time_per_output_token_seconds TPOT histogram.
# TYPE vllm:request_time_per_output_token_seconds histogram
vllm:request_time_per_output_token_seconds_bucket{le=\"0.01\"} 50
vllm:request_time_per_output_token_seconds_bucket{le=\"0.05\"} 80
vllm:request_time_per_output_token_seconds_bucket{le=\"0.1\"} 95
vllm:request_time_per_output_token_seconds_bucket{le=\"0.5\"} 99
vllm:request_time_per_output_token_seconds_bucket{le=\"+Inf\"} 100
vllm:request_time_per_output_token_seconds_sum 2.0
vllm:request_time_per_output_token_seconds_count 100.0
";
        let parsed = parse_prometheus_text(body).expect("parse");
        let buckets = parsed
            .histograms
            .get("vllm_request_time_per_output_token_seconds")
            .expect("tpot histogram captured under normalized key");
        let p50 = percentile(buckets, 0.5).expect("p50") * 1000.0;
        let p95 = percentile(buckets, 0.95).expect("p95") * 1000.0;
        let p99 = percentile(buckets, 0.99).expect("p99") * 1000.0;
        assert!(p50 < p95 && p95 < p99, "p50 {p50} < p95 {p95} < p99 {p99}");

        let sum = parsed
            .counters
            .get("vllm_request_time_per_output_token_seconds_sum")
            .copied()
            .expect("sum");
        let count = parsed
            .counters
            .get("vllm_request_time_per_output_token_seconds_count")
            .copied()
            .expect("count");
        let tpot_ms = (sum / count) * 1000.0;
        assert!((tpot_ms - 20.0).abs() < 1e-6, "tpot {tpot_ms} ≈ 20ms");
    }

    /// vLLM v0.6 names the metric `time_per_output_token_seconds` (no
    /// `request_` prefix). The resolver must fall back to it when the v1
    /// name is absent.
    #[test]
    fn tpot_falls_back_to_v06_metric_name() {
        let body = "\
# HELP vllm:time_per_output_token_seconds TPOT histogram.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le=\"0.05\"} 10
vllm:time_per_output_token_seconds_bucket{le=\"+Inf\"} 10
vllm:time_per_output_token_seconds_sum 0.3
vllm:time_per_output_token_seconds_count 10.0
";
        let parsed = parse_prometheus_text(body).expect("parse");
        assert!(
            !parsed
                .histograms
                .contains_key("vllm_request_time_per_output_token_seconds"),
            "v1 name must be absent in this fixture"
        );
        assert!(
            parsed
                .histograms
                .contains_key("vllm_time_per_output_token_seconds"),
            "v0.6 name must be present so the resolver falls back to it"
        );
        let sum = parsed
            .counters
            .get("vllm_time_per_output_token_seconds_sum")
            .copied()
            .expect("v0.6 sum");
        let count = parsed
            .counters
            .get("vllm_time_per_output_token_seconds_count")
            .copied()
            .expect("v0.6 count");
        let tpot_ms = (sum / count) * 1000.0;
        assert!((tpot_ms - 30.0).abs() < 1e-6, "tpot {tpot_ms} ≈ 30ms");
    }

    /// Integration check: the warmup tracker baselines after the first
    /// observation, and percentiles computed from the second `/metrics` body
    /// reflect *only* the post-baseline observations — proving that the slow
    /// first inference does not pollute steady-state percentiles.
    #[test]
    fn warmup_tracker_excludes_first_observation_from_percentiles() {
        use super::super::warmup::WarmupTracker;

        // Body 1: a single slow observation in the (1.0, +Inf] bucket.
        let body_warmup = "\
# HELP vllm:time_to_first_token_seconds TTFT histogram.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le=\"0.05\"} 0
vllm:time_to_first_token_seconds_bucket{le=\"0.1\"} 0
vllm:time_to_first_token_seconds_bucket{le=\"0.5\"} 0
vllm:time_to_first_token_seconds_bucket{le=\"1.0\"} 0
vllm:time_to_first_token_seconds_bucket{le=\"+Inf\"} 1
vllm:time_to_first_token_seconds_sum 8.0
vllm:time_to_first_token_seconds_count 1.0
";
        // Body 2: 100 fast observations all in [0, 0.05] on top of the warmup.
        let body_steady = "\
# HELP vllm:time_to_first_token_seconds TTFT histogram.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le=\"0.05\"} 100
vllm:time_to_first_token_seconds_bucket{le=\"0.1\"} 100
vllm:time_to_first_token_seconds_bucket{le=\"0.5\"} 100
vllm:time_to_first_token_seconds_bucket{le=\"1.0\"} 100
vllm:time_to_first_token_seconds_bucket{le=\"+Inf\"} 101
vllm:time_to_first_token_seconds_sum 9.0
vllm:time_to_first_token_seconds_count 101.0
";

        // Body 0: simulates the first poll right after the dashboard attaches
        // — no requests yet. The tracker captures count=0 as its initial
        // cursor here; it transitions to Active only once the cursor advances
        // by `skip_requests`.
        let body_idle = "\
# HELP vllm:time_to_first_token_seconds TTFT histogram.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le=\"0.05\"} 0
vllm:time_to_first_token_seconds_bucket{le=\"0.1\"} 0
vllm:time_to_first_token_seconds_bucket{le=\"0.5\"} 0
vllm:time_to_first_token_seconds_bucket{le=\"1.0\"} 0
vllm:time_to_first_token_seconds_bucket{le=\"+Inf\"} 0
vllm:time_to_first_token_seconds_sum 0.0
vllm:time_to_first_token_seconds_count 0.0
";

        let mut tracker = WarmupTracker::new(1);

        let parsed_idle = parse_prometheus_text(body_idle).expect("parse idle");
        let out_idle = tracker.observe(&parsed_idle);
        assert!(out_idle.warming_up);
        assert!(!out_idle.just_transitioned);

        let parsed_warmup = parse_prometheus_text(body_warmup).expect("parse warmup");
        let out_warmup = tracker.observe(&parsed_warmup);
        // After the warmup request lands, the tracker baselines and emits
        // warming_up=false, just_transitioned=true. The adjusted histogram
        // contains (current - baseline) where current == baseline → all zeros.
        assert!(!out_warmup.warming_up);
        assert!(out_warmup.just_transitioned);

        let parsed_steady = parse_prometheus_text(body_steady).expect("parse steady");
        let out_steady = tracker.observe(&parsed_steady);
        assert!(!out_steady.warming_up);
        assert!(!out_steady.just_transitioned);

        let buckets = out_steady
            .adjusted
            .histograms
            .get("vllm_time_to_first_token_seconds")
            .expect("histogram");
        let p50 = percentile(buckets, 0.5).expect("p50") * 1000.0;
        let p95 = percentile(buckets, 0.95).expect("p95") * 1000.0;
        // All 100 post-baseline observations live in [0, 0.05]; p50 and p95
        // must land inside that bucket. Without the tracker the slow warmup
        // observation would push p99 (and the +Inf overflow) into the tail.
        assert!(p50 <= 50.0, "p50 {p50} should be in fast bucket (<=50ms)");
        assert!(p95 <= 50.0, "p95 {p95} should be in fast bucket (<=50ms)");
        // Sum and count deltas: 1.0 sum across 100 fast observations.
        let sum = out_steady
            .adjusted
            .counters
            .get("vllm_time_to_first_token_seconds_sum")
            .copied()
            .expect("sum delta");
        let count = out_steady
            .adjusted
            .counters
            .get("vllm_time_to_first_token_seconds_count")
            .copied()
            .expect("count delta");
        assert!((sum - 1.0).abs() < 1e-9, "sum delta {sum}");
        assert!((count - 100.0).abs() < 1e-9, "count delta {count}");
    }

    /// The frontend recomputes goodput from histogram buckets the backend
    /// ships in `EngineMetrics`. This test mirrors the `buckets_for` closure
    /// in `get_metrics()` against a representative `/metrics` body and
    /// asserts the wire-shape contract: cumulative counts preserved, +Inf
    /// replaced with `f64::MAX`, and the buckets serialize to plain JSON.
    #[test]
    fn ttft_buckets_replace_infinity_with_f64_max() {
        let body = "\
# HELP vllm:time_to_first_token_seconds TTFT histogram.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le=\"0.05\"} 50
vllm:time_to_first_token_seconds_bucket{le=\"0.1\"} 80
vllm:time_to_first_token_seconds_bucket{le=\"+Inf\"} 100
vllm:time_to_first_token_seconds_sum 12.0
vllm:time_to_first_token_seconds_count 100.0
";
        let parsed = parse_prometheus_text(body).expect("parse");
        let raw = parsed
            .histograms
            .get("vllm_time_to_first_token_seconds")
            .expect("histogram");
        let buckets: Vec<HistogramBucket> = raw
            .iter()
            .map(|&(le, cum)| HistogramBucket {
                le_seconds: if le.is_finite() { le } else { f64::MAX },
                cumulative_count: cum,
            })
            .collect();
        assert_eq!(buckets.len(), 3);
        assert!((buckets[0].le_seconds - 0.05).abs() < 1e-9);
        assert!((buckets[0].cumulative_count - 50.0).abs() < 1e-9);
        assert!((buckets[1].le_seconds - 0.1).abs() < 1e-9);
        assert_eq!(buckets[2].le_seconds, f64::MAX);
        assert!((buckets[2].cumulative_count - 100.0).abs() < 1e-9);

        // The wire format must be valid JSON — non-finite floats would break
        // `JSON.parse` on the frontend.
        let json = serde_json::to_string(&buckets).expect("serialize");
        assert!(
            !json.contains("inf") && !json.contains("Inf") && !json.contains("NaN"),
            "wire format must not contain non-finite tokens: {json}"
        );
    }

    /// Empty histograms produce `None`, matching the warmup/no-traffic case
    /// where the frontend should fall back to the backend `*_goodput_pct`.
    #[test]
    fn ttft_buckets_none_when_metric_absent() {
        let body = "# Empty body, no histograms.\n";
        let parsed = parse_prometheus_text(body).expect("parse");
        assert!(!parsed
            .histograms
            .contains_key("vllm_time_to_first_token_seconds"));
    }

    /// Cumulative token counters flow from the parser to the new
    /// `total_prompt_tokens` / `total_generation_tokens` fields. The adapter
    /// reads them off `parsed.counters` and casts f64 -> u64, so assert the
    /// same boundary the struct fields are populated from.
    #[test]
    fn token_totals_roundtrip_from_metrics_body() {
        let body = "\
# HELP vllm:prompt_tokens_total Prompt tokens counter.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 123456.0
# HELP vllm:generation_tokens_total Generation tokens counter.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 7890123.0
";
        let parsed = parse_prometheus_text(body).expect("parse");
        let prompt = parsed
            .counters
            .get("vllm_prompt_tokens_total")
            .copied()
            .map(|v| v as u64);
        let gen = parsed
            .counters
            .get("vllm_generation_tokens_total")
            .copied()
            .map(|v| v as u64);
        assert_eq!(prompt, Some(123_456));
        assert_eq!(gen, Some(7_890_123));
    }

    /// `prefix_cache_queries_total` is a pass-through lifetime counter read
    /// off `parsed.counters` and cast f64 -> u64 (same boundary as the token
    /// totals, ungated by warmup). Assert it propagates from the metrics body.
    #[test]
    fn prefix_cache_queries_roundtrip_from_metrics_body() {
        let body = "\
# HELP vllm:prefix_cache_queries Prefix cache queries counter.
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total 654321.0
# HELP vllm:prefix_cache_hits Prefix cache hits counter.
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total 123456.0
";
        let parsed = parse_prometheus_text(body).expect("parse");
        let queries = parsed
            .counters
            .get("vllm_prefix_cache_queries_total")
            .copied()
            .map(|v| v as u64);
        assert_eq!(queries, Some(654_321));
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

    /// vLLM's prometheus_client appends `_total` to the speculative-decoding
    /// counters. Assert the parser captures them under the normalized
    /// `vllm_spec_decode_*_total` keys the adapter reads.
    #[test]
    fn spec_decode_counters_roundtrip_from_metrics_body() {
        let body = "\
# HELP vllm:spec_decode_num_draft_tokens Cumulative drafted tokens.
# TYPE vllm:spec_decode_num_draft_tokens counter
vllm:spec_decode_num_draft_tokens_total 1000.0
# HELP vllm:spec_decode_num_accepted_tokens Cumulative accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens counter
vllm:spec_decode_num_accepted_tokens_total 720.0
# HELP vllm:spec_decode_num_drafts Cumulative draft attempts.
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts_total 240.0
";
        let parsed = parse_prometheus_text(body).expect("parse");
        let draft = parsed
            .counters
            .get("vllm_spec_decode_num_draft_tokens_total")
            .copied();
        let accepted = parsed
            .counters
            .get("vllm_spec_decode_num_accepted_tokens_total")
            .copied();
        let drafts = parsed
            .counters
            .get("vllm_spec_decode_num_drafts_total")
            .copied();
        assert_eq!(draft, Some(1000.0));
        assert_eq!(accepted, Some(720.0));
        assert_eq!(drafts, Some(240.0));
        // Derived signals computed from the parsed counters.
        assert_eq!(spec_acceptance_rate(accepted, draft), Some(72.0));
        assert_eq!(spec_mean_acceptance_length(accepted, drafts), Some(3.0));
    }

    /// A metrics body without any speculative-decoding lines must leave every
    /// spec-decode counter absent, so the adapter reports `None` and the
    /// frontend hides the section.
    #[test]
    fn spec_decode_counters_absent_when_not_configured() {
        let body = "\
# HELP vllm:generation_tokens Generated tokens.
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total 42.0
";
        let parsed = parse_prometheus_text(body).expect("parse");
        assert!(!parsed
            .counters
            .contains_key("vllm_spec_decode_num_draft_tokens_total"));
        assert!(!parsed
            .counters
            .contains_key("vllm_spec_decode_num_draft_tokens"));
    }
}
