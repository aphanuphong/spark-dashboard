import { useMemo } from 'react'
import { useLatestSnapshot, useMetricSeries } from '@/hooks/useMetricsStore'
import { usePageSelection } from '@/hooks/usePageSelection'
import { resolveHardware } from '@/lib/dashboard/hardwareSource'
import { remoteSeries, type DataPoint } from '@/lib/metricsHistoryStore'
import type { TimeWindow } from '@/types/events'
import type { MetricsSnapshot } from '@/types/metrics'
import type { GpuPanelResolution } from './useGpuPanel'

/**
 * The hardware half of the page's view, shared by every host-wide panel.
 *
 * `hardwareSource.ts` stores *which machine* the page reads (this one, or a
 * mirrored peer); `resolveHardware` answers what that means against the live
 * snapshot; this hook is where panels consume the answer — as a snapshot to
 * read current values from and a series key to chart the same machine's
 * history. Substitution is prohibited, exactly as with engine bindings: a peer
 * that is down or not mirrored is shown as what it is, never as this box's
 * numbers under a title that implied another machine.
 */
export type HardwarePanelResolution =
  /** No snapshot yet; neither machine is confirmed. */
  | { status: 'waiting' }
  /**
   * The page's machine, resolved. `seriesFor` maps a host-wide metric name to
   * this machine's history series — the bare names for this host, peer-keyed
   * ones for a mirror. `peer` is set only when the machine is one: the panels
   * that draw a device name say whose hardware it is with it.
   */
  | { status: 'resolved'; snapshot: MetricsSnapshot; seriesFor: (metric: string) => string; peer?: { url: string; label: string } }
  /** The page points at a peer this dashboard mirrors no connection to. */
  | { status: 'peer-not-mirrored'; url: string }
  /** The peer is mirrored but not currently reachable. */
  | { status: 'peer-down'; url: string }

export function useHardwarePanel(): HardwarePanelResolution {
  const snapshot = useLatestSnapshot()
  const { hardware } = usePageSelection()

  return useMemo(() => {
    const resolution = resolveHardware(hardware, snapshot)

    switch (resolution.status) {
      case 'local':
        // An unresolved local read is the same waiting state the panels had
        // before this hook existed: absence means this machine, so a missing
        // snapshot here is simply no data yet.
        return snapshot ? { status: 'resolved', snapshot, seriesFor: (metric) => metric } : { status: 'waiting' }
      case 'remote': {
        const url = resolution.host.url
        return {
          status: 'resolved',
          snapshot: resolution.host.data!,
          seriesFor: (metric) => remoteSeries(url, metric),
          peer: { url, label: resolution.host.label },
        }
      }
      case 'waiting':
        return { status: 'waiting' }
      case 'not-mirrored':
        return { status: 'peer-not-mirrored', url: resolution.url }
      case 'down':
        return { status: 'peer-down', url: resolution.url }
    }
  }, [snapshot, hardware])
}

/**
 * A host-wide hardware panel's whole subscription in one call, on the model of
 * `useGpuPanelSeries`: resolve against the page's hardware source, then chart
 * that machine's series. While unresolved, the bare local key keeps the series
 * subscription alive until the first snapshot names the real machine.
 */
export function useHardwarePanelSeries(
  metric: string,
  window: TimeWindow,
): { resolution: HardwarePanelResolution; data: DataPoint[] } {
  const resolution = useHardwarePanel()
  const series = resolution.status === 'resolved' ? resolution.seriesFor(metric) : metric
  const data = useMetricSeries(series, window)
  return { resolution, data }
}

/**
 * The device label for a panel reading through this hook. This machine keeps
 * its own name exactly as it did before the choice existed. A peer's device
 * keeps its name too — dgx1's GB10 is still a GB10 — but gains the peer's
 * label, because a page of panels that are not this box should say so on the
 * title row and not only in the page config someone has to open.
 */
export function hardwareDevice(
  name: string | null | undefined,
  resolution: HardwarePanelResolution,
): string | null {
  const base = name ?? null
  if (resolution.status !== 'resolved' || !resolution.peer) return base
  return base ? `${base} · ${resolution.peer.label}` : resolution.peer.label
}

/**
 * The GPU counterpart of `hardwareDevice`, for panels whose resolution comes
 * from `useGpuPanel` rather than this module.
 */
export function hardwareDeviceForGpu(
  name: string | null | undefined,
  resolution: GpuPanelResolution,
): string | null {
  const base = name ?? null
  if (resolution.status !== 'resolved') return base
  if (!resolution.peer) return base
  return resolution.peer.label ? `${base ? `${base} · ` : ''}${resolution.peer.label}` : base
}
