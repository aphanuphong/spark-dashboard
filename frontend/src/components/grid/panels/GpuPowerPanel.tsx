import { ArcGauge } from '@/components/gauges/ArcGauge'
import { HBar } from '@/components/gauges/HBar'
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart'
import { computePowerScale, powerPeak } from '@/lib/gpuPower'
import { THRESHOLDS } from '@/lib/theme'
import { gpuIndexOf } from '@/lib/identity'
import { gpuLabel, gpuPanelDevice } from './gpuLabel'
import { gpuSeriesColor } from './gpuColor'
import { GpuPanelNotice } from './PanelNotice'
import { HardwarePanelBody } from './HardwarePanelBody'
import { MultiGpuPanelBody } from './MultiGpuPanelBody'
import { useGpuPanelSeriesAll } from './useGpuPanel'
import type { PanelContentProps } from '../panelRegistry'
import type { DataPoint } from '@/lib/metricsHistoryStore'

/** One GPU's rendering for the aggregate power panel: its percent of scale,
 *  whether against the hardware cap or the peak observed in this window. */
function powerEntry(
  watts: number | null,
  limit: number | null,
  data: DataPoint[],
  label: string,
) {
  const { percent } = computePowerScale(watts, limit, powerPeak(data, watts))
  const display = watts !== null ? Math.round(watts) : 0
  return { percent, display, label }
}

/**
 * GPU power draw. The gauge scales against the hardware limit when the GPU
 * reports one, and against the observed peak in this panel's own history
 * window otherwise (unified-memory SoCs expose no cap — see `lib/gpuPower`).
 *
 * Single-GPU machines and pinned panels get one gauge and one line. A
 * multi-GPU machine — archml with its two 5090s — shows every GPU at once:
 * stacked gauges and one chart carrying a line per GPU, each line keeping its
 * panel-independent colour.
 */
export function GpuPowerPanel({ panel }: PanelContentProps) {
  const aggregate = useGpuPanelSeriesAll(panel, 'gpuPower')
  const { resolution } = aggregate
  if (resolution.status !== 'resolved') return <GpuPanelNotice resolution={resolution} />


  if (aggregate.view === 'all') {
    return (
      <MultiGpuPanelBody
        device={gpuPanelDevice(aggregate.gpus.map((g) => g.gpu), resolution)}
        entries={aggregate.gpus.map(({ gpu }, i) => {
          const { percent, display, label } = powerEntry(
            gpu.power_watts,
            gpu.power_limit_watts,
            aggregate.perGpu[i],
            `GPU ${gpuIndexOf(gpu)}`,
          )
          return {
            compact: (
              <HBar
                value={percent}
                label={label}
                unit="W"
                thresholds={THRESHOLDS.gpuPower}
                displayValue={display}
              />
            ),
            gauge: (sizePx: number) => (
              <ArcGauge
                value={percent}
                label={label}
                unit="W"
                thresholds={THRESHOLDS.gpuPower}
                displayValue={display}
                size={sizePx}
              />
            ),
          }
        })}
        chart={
          <TimeSeriesChart
            series={aggregate.gpus.map(({ gpu }, i) => ({
              label: `GPU ${gpuIndexOf(gpu)}`,
              data: aggregate.perGpu[i],
              color: gpuSeriesColor(i),
            }))}
            unit="W"
          />
        }
      />
    )
  }

  const { gpu } = resolution
  const { percent, display, label } = powerEntry(
    gpu.power_watts,
    gpu.power_limit_watts,
    aggregate.data,
    gpuLabel(resolution, 'GPU Power'),
  )
  return (
    <HardwarePanelBody
      device={gpuPanelDevice([resolution.gpu], resolution)}
      compact={
        <HBar
          value={percent}
          label={label}
          unit="W"
          thresholds={THRESHOLDS.gpuPower}
          displayValue={display}
        />
      }
      gauge={(sizePx) => (
        <ArcGauge
          value={percent}
          label={label}
          unit="W"
          thresholds={THRESHOLDS.gpuPower}
          displayValue={display}
          size={sizePx}
        />
      )}
      chart={<TimeSeriesChart data={aggregate.data} unit="W" seriesLabel="Power" />}
    />
  )
}
