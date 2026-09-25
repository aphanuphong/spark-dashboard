/**
 * The mirrored peers a page's hardware source could name, taken from the live
 * snapshot.
 *
 * Split out of the panel plumbing because both the page-config control and any
 * later peer-binding UI need the same projection of `snapshot.remote`: the
 * address is identity (what `HardwareSource` stores), the label is what the
 * control displays, and a peer that is listed but down is still a machine the
 * page may point at — reconnect is automatic.
 */

import type { MetricsSnapshot } from '@/types/metrics'
import type { MirroredPeer } from './hardwareSource'

/** The peers this host mirrors, in snapshot order. */
export function mirroredPeers(
  snapshot: Pick<MetricsSnapshot, 'remote'> | null,
): MirroredPeer[] {
  return (snapshot?.remote ?? []).map((host) => ({ url: host.url, label: host.label }))
}
