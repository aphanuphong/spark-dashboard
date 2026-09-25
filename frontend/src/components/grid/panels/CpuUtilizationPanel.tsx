import { ArcGauge } from '@/components/gauges/ArcGauge'
import { HBar } from '@/components/gauges/HBar'
import { CoreHeatmap } from '@/components/charts/CoreHeatmap'
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart'
import { THRESHOLDS } from '@/lib/theme'
import { PanelNotice, HardwarePanelNotice } from './PanelNotice'
import { HardwarePanelBody } from './HardwarePanelBody'
import { hardwareDevice, useHardwarePanelSeries } from './useHardwarePanel'
import type { PanelContentProps } from '../panelRegistry'

/**
 * Aggregate CPU utilization, with the per-core heatmap when the panel is tall
 * enough for the full rendering. Host-wide, so nothing needs binding — which
 * machine it is host-wide *on* is the page's hardware source.
 */
export function CpuUtilizationPanel({ panel }: PanelContentProps) {
  const { resolution, data } = useHardwarePanelSeries('cpuAggregate', panel.window)
  if (resolution.status === 'peer-not-mirrored' || resolution.status === 'peer-down') {
    return <HardwarePanelNotice resolution={resolution} />
  }
  if (resolution.status !== 'resolved') return <PanelNotice>Waiting for metrics</PanelNotice>

  const { cpu } = resolution.snapshot

  return (
    <HardwarePanelBody
      device={hardwareDevice(cpu.name, resolution)}
      compact={
        <HBar value={cpu.aggregate_percent} label="CPU" unit="%" thresholds={THRESHOLDS.cpuUsage} />
      }
      gauge={(sizePx) => (
        <ArcGauge
          value={cpu.aggregate_percent}
          label="CPU"
          unit="%"
          thresholds={THRESHOLDS.cpuUsage}
          size={sizePx}
        />
      )}
      chart={
        <TimeSeriesChart data={data} yDomain={[0, 100]} unit="%" seriesLabel="CPU" />
      }
      below={cpu.per_core.length > 0 ? <CoreHeatmap cores={cpu.per_core} /> : undefined}
    />
  )
}
