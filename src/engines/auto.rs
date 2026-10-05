//! Auto-detecting engine adapter.
//!
//! Wraps the vLLM and DGPP adapters and picks between them per poll by
//! sniffing the `/metrics` response: DGPP serves a JSON document with
//! `scheduler` / `service` / `prefix_cache` sections, vLLM a Prometheus text
//! exposition. This is what lets one endpoint (e.g. dgx1.rt-ctrl.com:8000)
//! host either engine over time — the dashboard follows whatever is live
//! without a config change or restart, which matters because swapping the
//! engine necessarily takes the endpoint down for a while.
//!
//! Once the flavor is known, the adapter reports it as its engine type, so
//! the dashboard renders the tile exactly as it would for a pinned engine.
//! Before the first successful sniff it reports `Auto`.

use super::dgpp::DgppAdapter;
use super::metadata::ModelMetadata;
use super::vllm::VllmAdapter;
use super::{EngineAdapter, EngineMetrics, EngineStatus, EngineType, ModelResolution};
use async_trait::async_trait;
use std::sync::Mutex;
use std::time::Duration;

pub struct AutoAdapter {
    client: reqwest::Client,
    endpoint: String,
    /// Optional bearer token for an auth-gated endpoint. Never printed.
    api_key: Option<String>,
    /// `/v1/models` resolution + HuggingFace enrichment. Both wrapped engines
    /// speak the same OpenAI-style models endpoint, so the auto adapter owns
    /// one shared resolver instead of duplicating cache state per flavor.
    metadata: ModelMetadata,
    vllm: VllmAdapter,
    dgpp: DgppAdapter,
    /// Engine flavor observed at the endpoint; `Auto` until the first
    /// successful `/metrics` sniff. Reported as the snapshot's engine type.
    /// A plain mutex: guarded sections never await.
    flavor: Mutex<EngineType>,
}

impl AutoAdapter {
    pub fn new(
        client: reqwest::Client,
        endpoint: String,
        served_model: Option<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            vllm: VllmAdapter::new(
                client.clone(),
                endpoint.clone(),
                served_model.clone(),
                api_key.clone(),
            ),
            dgpp: DgppAdapter::new(
                client.clone(),
                endpoint.clone(),
                served_model.clone(),
                api_key.clone(),
            ),
            client,
            endpoint,
            api_key,
            metadata: ModelMetadata::new(served_model),
            flavor: Mutex::new(EngineType::Auto),
        }
    }

    /// Attach the bearer token when one is configured. No-op otherwise.
    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => rb.bearer_auth(key),
            None => rb,
        }
    }
}

/// Whether a `/metrics` body is a DGPP JSON document.
///
/// The endpoint's content type is not trusted (proxies rewrite it); the body
/// decides. DGPP's document is JSON with at least one of its well-known
/// sections at the top level; vLLM serves Prometheus text, which fails JSON
/// parsing and routes to the vLLM path.
fn looks_like_dgpp_json(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .is_some_and(|v| {
            v.get("scheduler").is_some() || v.get("service").is_some() || v.get("prefix_cache").is_some()
        })
}

#[async_trait]
impl EngineAdapter for AutoAdapter {
    fn engine_type(&self) -> EngineType {
        self.flavor.lock().expect("flavor mutex").clone()
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    async fn health_check(&self) -> EngineStatus {
        // Both engines answer GET /health with a plain success status.
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
        // One fetch per poll; both flavor paths parse the same body. The
        // wrapped adapters each also expose a fetch-and-parse `get_metrics`,
        // but delegating to those would hit the endpoint twice per tick and
        // diff two different snapshots.
        let body = self
            .auth(
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
            .ok()?;

        let detected = if looks_like_dgpp_json(&body) {
            EngineType::Dgpp
        } else {
            EngineType::Vllm
        };

        let flipped = {
            let mut flavor = self.flavor.lock().expect("flavor mutex");
            let flipped = *flavor != detected;
            if flipped {
                // Counter snapshots, running averages, and warmup baselines
                // all belong to the engine that was live before. Diffing them
                // across a swap would surface one bogus rate (or a bogus
                // warmup gate), so both wrapped adapters start clean on a
                // flavor change. On the very first sniff this is a no-op on
                // fresh state.
                tracing::info!(
                    endpoint = %self.endpoint,
                    "engine flavor at endpoint is now {}",
                    detected
                );
            }
            *flavor = detected.clone();
            flipped
        };
        if flipped {
            self.vllm.reset_derived_state().await;
            self.dgpp.reset_derived_state().await;
        }

        match detected {
            EngineType::Dgpp => self.dgpp.process_metrics_text(&body).await,
            _ => self.vllm.process_metrics_text(&body).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dgpp_json_is_recognized_by_its_sections() {
        assert!(looks_like_dgpp_json(
            r#"{"scheduler": {"active": 0}, "service": {}, "prefix_cache": {}}"#
        ));
        assert!(looks_like_dgpp_json(r#"{"scheduler": {"active": 1}}"#));
        assert!(looks_like_dgpp_json(r#"{"prefix_cache": {"hits": 1}}"#));
    }

    #[test]
    fn non_dgpp_bodies_route_to_the_vllm_path() {
        // Prometheus text does not parse as JSON.
        assert!(!looks_like_dgpp_json("# HELP vllm:x x\nvllm:x 1.0\n"));
        // JSON without any DGPP section is not DGPP either (some other
        // engine's JSON metrics endpoint must not hijack the DGPP path).
        assert!(!looks_like_dgpp_json(r#"{"tokens": 1}"#));
        assert!(!looks_like_dgpp_json(""));
    }
}
