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

/**
 * What an *aggregate* GPU panel renders: every GPU the machine has, at once.
 *
 * A following panel (binding `follow`, or one pinned to the page's GPU) that
 * lands on a multi-GPU machine shows all of its GPUs together — two gauges and
 * two chart lines on a two-GPU box like archml's, one of each on a single-GPU
 * box or a mirrored peer, where that is simply what the machine is. A panel
 * pinned to one GPU keeps showing exactly that GPU: a pin is an explicit ask,
 * and an aggregate view would bury the target the operator retitled the panel
 * for. This is also what makes DGX1 correct: it has one GPU, so pointing a
 * multi-GPU box's utilization panel at it narrows to one gauge and one line
 * without the panel knowing or caring.
 */
export type GpuAggregate =
  /** Show exactly what `useGpuPanelSeries` says: the machine is a single-GPU
   *  one, or this panel's binding pinned one GPU. */
  | { view: 'one'; resolution: GpuPanelResolution; data: DataPoint[] }
  /** Every GPU's current value plus one chart series each, in GPU order. */
  | {
      view: 'all'
      resolution: Extract<GpuPanelResolution, { status: 'resolved' }>
      gpus: Array<{
        gpu: GpuMetrics
        /** The history series key for this GPU's metric. */
        series: string
      }>
      /** Charted data per GPU, index-aligned with `gpus`. */
      perGpu: DataPoint[][]
    }

/**
 * `useGpuPanelSeries` on aggregate terms: the panel subscribes to every GPU's
 * series when there is more than one to show. Four subscriptions, fixed —
 * hook order cannot depend on how many GPUs the machine happens to have, and
 * the rendering ladder names colors for four GPUs anyway.
 */
export function useGpuPanelSeriesAll(
  panel: DashboardPanel,
  metric: GpuSeriesMetric,
): GpuAggregate {
  const resolution = useGpuPanel(panel)
  const resolved = resolution.status === 'resolved'
  const aggregate = resolved && resolution.multiGpu && panel.binding.kind !== 'gpu'

  const gpus = resolved ? snapshotGpus(resolution.snapshot) : []
  // `seriesFor` builds the key of the single resolved GPU; a whole-machine
  // view needs each GPU's own key, so a following panel builds them itself.
  // A mirrored peer keys all of its GPUs, including its first one.
  const keys = resolved
    ? aggregate
      ? resolution.peer
        ? gpus.map((gpu) => remoteGpuSeries(resolution.peer!.url, metric, gpuIndexOf(gpu)))
        : gpus.map((gpu) => gpuSeries(metric, gpuIndexOf(gpu), true))
      : [resolution.seriesFor(metric)]
    : []

  const d0 = useMetricSeries(keys[0] ?? idleSeries(metric), panel.window)
  const d1 = useMetricSeries(keys[1] ?? idleSeries(metric), panel.window)
  const d2 = useMetricSeries(keys[2] ?? idleSeries(metric), panel.window)
  const d3 = useMetricSeries(keys[3] ?? idleSeries(metric), panel.window)
  const perGpu = [d0, d1, d2, d3]

  if (aggregate) {
    const shown = Math.min(gpus.length, perGpu.length)
    return {
      view: 'all',
      resolution,
      gpus: gpus.slice(0, shown).map((gpu, i) => ({ gpu, series: keys[i] })),
      perGpu: perGpu.slice(0, shown),
    }
  }
  return { view: 'one', resolution, data: perGpu[0] ?? [] }
}

/**
 * A series that exists in no buffer. Subscribing costs the store nothing and
 * reading it returns nothing — the slot draws no line — which is how unused
 * hook positions pass through a panel on a machine with fewer GPUs.
 */
function idleSeries(metric: GpuSeriesMetric): string {
  return `~unused~${metric}`
}

