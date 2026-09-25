//! Remote hardware mirroring over a peer spark-dashboard's WebSocket.
//!
//! A peer spark-dashboard broadcasts its own `MetricsSnapshot` JSON on `/ws`.
//! Rather than scraping exporters on the remote host, we consume that stream
//! verbatim — the two hosts speak the same wire format by construction — and
//! graft the latest snapshot onto this host's broadcast under `remote[]`.
//!
//! Each target is a background task with its own reconnect loop so one dead
//! peer never disturbs the others, and the shared state keeps the target's
//! entry present (with `connected: false`) even while it is down, so panels
//! can show "connection lost" instead of vanishing.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::RwLock;
use tokio_tungstenite::tungstenite::Message;

use crate::metrics::RemoteSnapshot;

/// One peer dashboard to mirror. `url` may be `http(s)://` or `ws(s)://`; a
/// bare origin is upgraded and `/ws` appended.
#[derive(Clone, Debug)]
pub struct RemoteTarget {
    pub url: String,
    pub label: String,
}

/// Shared view of every target, keyed by its configured URL. The metrics
/// collector clones this into each broadcast snapshot.
pub type RemoteState = Arc<RwLock<HashMap<String, RemoteSnapshot>>>;

/// Spawn one mirror task per target. The map is seeded immediately so the
/// remote hosts appear in snapshots from the first tick, just not connected.
pub fn spawn_mirrors(targets: Vec<RemoteTarget>, state: RemoteState) {
    for target in targets {
        let state = Arc::clone(&state);
        if let Ok(mut map) = state.try_write() {
            map.insert(
                target.url.clone(),
                RemoteSnapshot {
                    url: target.url.clone(),
                    label: target.label.clone(),
                    connected: false,
                    data: None,
                },
            );
        }
        tokio::spawn(mirror_task(target, state));
    }
}

async fn mirror_task(target: RemoteTarget, state: RemoteState) {
    let ws_url = match normalize_ws_url(&target.url) {
        Some(url) => url,
        None => {
            tracing::error!(
                "Remote mirror {}: not a valid URL, disabling this target",
                target.url
            );
            return;
        }
    };

    loop {
        match tokio_tungstenite::connect_async(&ws_url).await {
            Ok((mut socket, _)) => {
                tracing::info!("Remote mirror connected: {}", ws_url);
                set_connected(&state, &target.url, true).await;
                loop {
                    match socket.next().await {
                        Some(Ok(Message::Text(text))) => match serde_json::from_str::<Value>(&text)
                        {
                            Ok(data) => {
                                let mut map = state.write().await;
                                if let Some(entry) = map.get_mut(&target.url) {
                                    entry.connected = true;
                                    entry.data = Some(data);
                                }
                            }
                            Err(err) => {
                                tracing::debug!(
                                    "Remote mirror {}: unparseable frame ({}), ignoring",
                                    target.url,
                                    err
                                );
                            }
                        },
                        // Close, error, or stream end: fall through to the
                        // reconnect arm below after marking disconnected.
                        Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                        // Pings/pongs/binaries carry nothing we store.
                        Some(Ok(_)) => {}
                    }
                }
                // Half-open peers that never send a Close frame still surface
                // through `next()` errors or the next write; try a clean close
                // so the peer logs a normal disconnect.
                let _ = socket.close(None).await;
            }
            Err(err) => {
                tracing::warn!("Remote mirror {}: connect failed ({err}), retrying in 5s", ws_url);
            }
        }
        set_connected(&state, &target.url, false).await;
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

async fn set_connected(state: &RemoteState, url: &str, connected: bool) {
    let mut map = state.write().await;
    if let Some(entry) = map.get_mut(url) {
        entry.connected = connected;
    }
}

/// `http(s)://host[:port]` → `ws(s)://host[:port]/ws`; `ws(s)://` URLs are
/// passed through untouched (the operator may be pointing at an explicit path).
fn normalize_ws_url(url: &str) -> Option<String> {
    if let Some(rest) = url.strip_prefix("http://") {
        Some(format!("ws://{rest}/ws"))
    } else if let Some(rest) = url.strip_prefix("https://") {
        Some(format!("wss://{rest}/ws"))
    } else if url.starts_with("ws://") || url.starts_with("wss://") {
        Some(url.to_owned())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_origin_becomes_ws_with_path() {
        assert_eq!(
            normalize_ws_url("http://dgx1:3000").unwrap(),
            "ws://dgx1:3000/ws"
        );
        assert_eq!(
            normalize_ws_url("https://peer.example").unwrap(),
            "wss://peer.example/ws"
        );
    }

    #[test]
    fn ws_urls_pass_through_verbatim() {
        assert_eq!(
            normalize_ws_url("ws://dgx1:3000/ws").unwrap(),
            "ws://dgx1:3000/ws"
        );
        assert_eq!(normalize_ws_url("wss://x/y").unwrap(), "wss://x/y");
    }

    #[test]
    fn non_urls_rejected() {
        assert!(normalize_ws_url("dgx1:3000").is_none());
        assert!(normalize_ws_url("ftp://nope").is_none());
    }
}
