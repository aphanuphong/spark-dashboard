import { NVIDIA_THEME } from '@/lib/theme'

/**
 * The chart colour for one GPU on a multi-GPU host.
 *
 * A panel that draws every GPU in one box has to say which line is which
 * twice over: the legend rows carry each GPU's own name, and the colours come
 * from this fixed ladder so the same GPU is the same colour in the utilization
 * chart as it is in the power and temperature charts. Reordering GPUs between
 * ticks must never recolor a line — index decides, nothing else.
 *
 * GPU 0 keeps the chart's primary green: a single-GPU machine draws with that
 * colour, and following a multi-GPU machine with one GPU is the common case.
 * Later GPUs take alternates that stay legible on both surfaces.
 */
export function gpuSeriesColor(index: number): string {
  if (index === 0) return NVIDIA_THEME.chartLine
  if (index === 1) return '#22d3ee'
  if (index === 2) return '#f59e0b'
  if (index === 3) return '#a78bfa'
  return index % 2 === 0 ? NVIDIA_THEME.chartLine : '#22d3ee'
}
