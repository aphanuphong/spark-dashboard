import { gpuIndexOf } from '@/lib/identity'
import type { GpuPanelResolution } from './useGpuPanel'
import { hardwareDeviceForGpu } from './useHardwarePanel'

/**
 * The gauge label for a resolved GPU panel. On a multi-GPU host it names the
 * GPU the panel resolved to — a panel's data and its label must agree, and
 * with several GPUs the metric name alone would not say whose numbers these
 * are. Single-GPU hosts keep the metric label the pre-grid dashboard used.
 */
export function gpuLabel(
  resolution: Extract<GpuPanelResolution, { status: 'resolved' }>,
  metricLabel: string,
): string {
  return resolution.multiGpu ? `GPU ${gpuIndexOf(resolution.gpu)}` : metricLabel
}

/**
 * The frame device row for a panel drawing the machine's GPUs together: the
 * models it reads, a twin pair named once (`NVIDIA GeForce RTX 5090`, not the
 * same words joined by a plus). Pass the one GPU the panel resolved to when
 * the view is not aggregate — then this is exactly what the device row said
 * before multi-GPU panels existed.
 */
export function gpuPanelDevice(
  gpus: readonly { name: string | null | undefined }[],
  resolution: GpuPanelResolution,
): string | null {
  const names = [...new Set(gpus.map((gpu) => gpu.name).filter((name) => name != null))]
  return hardwareDeviceForGpu(names.length > 0 ? names.join(' + ') : null, resolution)
}
