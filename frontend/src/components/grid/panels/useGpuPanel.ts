import { useMemo } from 'react'
import { useMetricSeries } from '@/hooks/useMetricsStore'
import { usePageSelection } from '@/hooks/usePageSelection'
import { resolveGpuBinding } from '@/lib/dashboard/bindings'
import { pageSelection } from '@/lib/dashboard/selection'
import { gpuIndexOf, snapshotGpus } from '@/lib/identity'
import { gpuSeries, remoteEventsSeries, remoteGpuSeries, type DataPoint, type GpuSeriesMetric } from '@/lib/metricsHistoryStore'
import type { DashboardPanel } from '@/lib/dashboard/schema'
import type { MetricsSnapshot, GpuMetrics } from '@/types/metrics'
import { useHardwarePanel } from './useHardwarePanel'

export type GpuPanelResolution =
  /** No snapshot has arrived yet; there are no GPUs to resolve against. */
  | { status: 'waiting' }
  | {
      status: 'resolved'
      /** The machine this GPU's values and series come from. */
      snapshot: MetricsSnapshot
      gpu: GpuMetrics
      multiGpu: boolean
      /** The history series key for this GPU's metric — per-GPU keys on
       *  multi-GPU hosts, the legacy un-prefixed keys on single-GPU ones. */
      seriesFor: (metric: GpuSeriesMetric) => string
      /** Set when the page reads a mirrored peer: the panels that draw a
       *  device name say whose GPU it is with it. */
      peer?: { url: string; label: string }
      /** The peer's GPU-event buffer, or undefined for this machine. */
      eventsSeries?: string
    }
  | { status: 'missing'; requested: string }
  | { status: 'unselected' }
  | { status: 'unreadable' }
  /** The page points at a peer this dashboard mirrors no connection to. */
  | { status: 'peer-not-mirrored'; url: string }
  /** The peer is mirrored but not currently reachable. */
  | { status: 'peer-down'; url: string }

/**
 * What a GPU panel renders on the page's machine: its binding resolved against
 * the selected machine's GPUs, plus the series-key vocabulary for its charts.
 *
 * A following panel resolves to the page-level GPU selection, which is the
 * primary GPU until the operator points the page somewhere else — so a page of
 * following panels moves to another GPU coherently, all at once. When the page
 * is pointed at a mirrored peer (`useHardwarePanel`), that peer's GPUs are the
 * ones resolved against and its history is what the charts draw — a machine
 * that cannot be read is reported, never substituted.
 *
 * Exported for the panels that bind to a GPU without charting one of its
 * series — the event list. Panels that do chart one take
 * `useGpuPanelSeries`, which resolves and subscribes together.
 */
export function useGpuPanel(panel: DashboardPanel): GpuPanelResolution {
  const hardware = useHardwarePanel()
  const { chosen } = usePageSelection()

  return useMemo(() => {
    if (hardware.status === 'peer-not-mirrored') return { status: 'peer-not-mirrored', url: hardware.url }
    if (hardware.status === 'peer-down') return { status: 'peer-down', url: hardware.url }
    if (hardware.status !== 'resolved') return { status: 'waiting' }
    const { snapshot } = hardware

    const gpus = snapshotGpus(snapshot)
    const resolution = resolveGpuBinding(
      panel.binding,
      gpus,
      pageSelection(snapshot, chosen).gpuIndex,
    )
    if (resolution.status !== 'resolved') return resolution

    const multiGpu = gpus.length > 1
    const index = gpuIndexOf(resolution.target)
    // On a mirrored peer every GPU is keyed, so there is no single-GPU
    // exception: the peer's first GPU is as much `gpu:0:…` as its fifth.
    const peer = hardware.peer
    return {
      status: 'resolved',
      snapshot,
      gpu: resolution.target,
      multiGpu,
      seriesFor: (metric: GpuSeriesMetric) =>
        peer ? remoteGpuSeries(peer.url, metric, index) : gpuSeries(metric, index, multiGpu),
      ...(peer ? { peer, eventsSeries: remoteEventsSeries(peer.url) } : {}),
    }
  }, [hardware, chosen, panel.binding])
}

/**
 * A GPU panel's whole subscription in one call: the resolved binding and the
 * chart data for `metric` over the panel's own window. Every hook lives in
 * here, above any caller's unresolved early return; while unresolved, the
 * legacy un-prefixed key keeps the series subscription alive until the first
 * snapshot names the real one.
 */
export function useGpuPanelSeries(
  panel: DashboardPanel,
  metric: GpuSeriesMetric,
): { resolution: GpuPanelResolution; data: DataPoint[] } {
  const resolution = useGpuPanel(panel)
  const series = resolution.status === 'resolved' ? resolution.seriesFor(metric) : metric
  const data = useMetricSeries(series, panel.window)
  return { resolution, data }
}
