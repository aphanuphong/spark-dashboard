import { ArcGauge } from '@/components/gauges/ArcGauge'
import { HBar } from '@/components/gauges/HBar'
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart'
import { formatGiB } from '@/lib/format'
import { memorySplit } from '@/lib/memorySplit'
import { PanelNotice, HardwarePanelNotice } from './PanelNotice'
import { HardwarePanelBody } from './HardwarePanelBody'
import { hardwareDevice, useHardwarePanelSeries } from './useHardwarePanel'
import type { PanelContentProps } from '../panelRegistry'

/**
 * Used share of the host's memory pool, split into the product's segments
 * (GPU, CPU, cache, free). Host-wide: unified-memory machines report one pool
 * shared with the GPU, which is why this panel does not bind to a GPU.
 * Host-wide also means it follows the page's hardware source — on a mirrored
 * peer, this is the peer's pool.
 */
export function MemoryPanel({ panel }: PanelContentProps) {
  const { resolution, data } = useHardwarePanelSeries('memoryUsedPercent', panel.window)
  if (resolution.status === 'peer-not-mirrored' || resolution.status === 'peer-down') {
    return <HardwarePanelNotice resolution={resolution} />
  }
  if (resolution.status !== 'resolved') return <PanelNotice>Waiting for metrics</PanelNotice>

  const { memory } = resolution.snapshot
  const { usedPercent, segments } = memorySplit(memory)
  // The pool's size is the panel's "which hardware": on a unified host it is
  // the one pool the GPU also draws from, which is why this panel is host-wide.
  const pool = formatGiB(memory.display_total_bytes ?? memory.total_bytes)

  return (
    <HardwarePanelBody
      device={hardwareDevice(
        memory.is_unified ? `${pool} Unified` : pool,
        resolution,
      )}
      compact={<HBar value={usedPercent} label="" unit="%" segments={segments} />}
      gauge={(sizePx) => (
        <ArcGauge value={usedPercent} label="" unit="%" segments={segments} size={sizePx} />
      )}
      chart={
        <TimeSeriesChart data={data} yDomain={[0, 100]} unit="%" seriesLabel="Used" />
      }
    />
  )
}
