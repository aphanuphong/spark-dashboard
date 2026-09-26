import { ArcGauge } from '@/components/gauges/ArcGauge'
import { HBar } from '@/components/gauges/HBar'
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart'
import { THRESHOLDS } from '@/lib/theme'
import { gpuIndexOf } from '@/lib/identity'
import { gpuLabel, gpuPanelDevice } from './gpuLabel'
import { gpuSeriesColor } from './gpuColor'
import { GpuPanelNotice } from './PanelNotice'
import { HardwarePanelBody } from './HardwarePanelBody'
import { MultiGpuPanelBody } from './MultiGpuPanelBody'
import { useGpuPanelSeriesAll } from './useGpuPanel'
import type { PanelContentProps } from '../panelRegistry'

/**
 * GPU temperature, colored by the product's thermal thresholds. Single-GPU
 * machines and pinned panels get one gauge and one line; a multi-GPU machine
 * shows every GPU together — stacked gauges and one chart with a line per GPU,
 * each keeping the same colour it has in the other GPU panels.
 */
export function GpuTemperaturePanel({ panel }: PanelContentProps) {
  const aggregate = useGpuPanelSeriesAll(panel, 'gpuTemp')
  const { resolution } = aggregate
  if (resolution.status !== 'resolved') return <GpuPanelNotice resolution={resolution} />


  if (aggregate.view === 'all') {
    return (
      <MultiGpuPanelBody
        device={gpuPanelDevice(aggregate.gpus.map((g) => g.gpu), resolution)}
        entries={aggregate.gpus.map(({ gpu }) => {
          const value = gpu.temperature_celsius ?? 0
          const label = `GPU ${gpuIndexOf(gpu)}`
          return {
            compact: <HBar value={value} label={label} unit="°C" thresholds={THRESHOLDS.gpuTemp} />,
            gauge: (sizePx: number) => (
              <ArcGauge
                value={value}
                label={label}
                unit="°C"
                thresholds={THRESHOLDS.gpuTemp}
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
            yDomain={[0, 100]}
            unit="°C"
          />
        }
      />
    )
  }

  const value = resolution.gpu.temperature_celsius ?? 0
  const label = gpuLabel(resolution, 'GPU Temp')
  return (
    <HardwarePanelBody
      device={gpuPanelDevice([resolution.gpu], resolution)}
      compact={<HBar value={value} label={label} unit="°C" thresholds={THRESHOLDS.gpuTemp} />}
      gauge={(sizePx) => (
        <ArcGauge value={value} label={label} unit="°C" thresholds={THRESHOLDS.gpuTemp} size={sizePx} />
      )}
      chart={
        <TimeSeriesChart data={aggregate.data} yDomain={[0, 100]} unit="°C" seriesLabel="Temp" />
      }
    />
  )
}
