import type { ReactNode } from 'react'
import { useElementSize } from '@/hooks/useElementSize'
import { usePanelDevice } from '../panelDevice'
import { gaugeSizePx, hardwarePanelMode } from './mode'

/** One GPU's rendering inside an aggregate panel. */
export interface GpuPanelEntry {
  /** The value-only rendering for a box too short to chart in. */
  compact: ReactNode
  /** The gauge for this GPU, given its square size in px. */
  gauge: (sizePx: number) => ReactNode
}

interface MultiGpuPanelBodyProps {
  /** The hardware these panels are reading — all of it, named together. */
  device?: string | null
  /** One entry per GPU, in GPU order. */
  entries: GpuPanelEntry[]
  /** The shared trend chart carrying one line per GPU. */
  chart: ReactNode
}

/**
 * The body of a GPU panel showing every GPU on the machine at once — the
 * aggregate counterpart of `HardwarePanelBody`.
 *
 * Same mode ladder as its single-GPU sibling (full / chart / compact), but
 * every element is per-GPU: in a full box the gauges stack in the gauge column
 * — each sized `gaugeSizePx(height, count)` so two 5090s on archml still get
 * legible dials — beside one chart drawing all of the lines together, which
 * is what makes the two GPUs comparable at a glance. In a compact box the
 * bars stack instead, and in a narrow box only the shared chart remains.
 */
export function MultiGpuPanelBody({ device, entries, chart }: MultiGpuPanelBodyProps) {
  const [ref, size] = useElementSize<HTMLDivElement>()
  const mode = hardwarePanelMode(size)
  usePanelDevice(device)

  return (
    <div
      ref={ref}
      // Same inline-height contract as HardwarePanelBody: the measured height
      // decides the mode wherever this renders, including the browser project.
      style={{ height: '100%' }}
      className="flex flex-col min-h-0 min-w-0 overflow-hidden"
    >
      {mode === 'compact' ? (
        <div className="flex-1 min-h-0 min-w-0 flex flex-col justify-center gap-1">
          {entries.map((entry, i) => (
            <div key={i} className="min-w-0">
              {entry.compact}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex-1 flex items-center gap-2 min-w-0 min-h-0 overflow-hidden">
          {mode === 'full' && (
            <div className="shrink-0 flex flex-col justify-center gap-1">
              {entries.map((entry, i) => (
                <div key={i} className="flex justify-center">
                  {entry.gauge(gaugeSizePx(size.height, entries.length))}
                </div>
              ))}
            </div>
          )}
          <div className="flex-1 min-w-0 h-full min-h-0">{chart}</div>
        </div>
      )}
    </div>
  )
}
