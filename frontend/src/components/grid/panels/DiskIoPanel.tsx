import { IoPanel } from './IoPanel'
import { HardwarePanelNotice } from './PanelNotice'
import { hardwareDevice, useHardwarePanelSeries } from './useHardwarePanel'
import type { PanelContentProps } from '../panelRegistry'

/** Disk read/write rates and their trend. Host-wide, so it follows the page's
 *  hardware source. */
export function DiskIoPanel({ panel }: PanelContentProps) {
  const read = useHardwarePanelSeries('diskRead', panel.window)
  const write = useHardwarePanelSeries('diskWrite', panel.window)
  const resolution = read.resolution
  if (resolution.status === 'peer-not-mirrored' || resolution.status === 'peer-down') {
    return <HardwarePanelNotice resolution={resolution} />
  }
  const snapshot = resolution.status === 'resolved' ? resolution.snapshot : null

  return (
    <IoPanel
      device={hardwareDevice(snapshot?.disk.name, resolution)}
      inbound={{
        tag: 'R',
        label: 'Read',
        color: '#76B900',
        rate: snapshot ? snapshot.disk.read_bytes_per_sec : null,
        data: read.data,
      }}
      outbound={{
        tag: 'W',
        label: 'Write',
        color: '#F59E0B',
        rate: snapshot ? snapshot.disk.write_bytes_per_sec : null,
        data: write.data,
      }}
    />
  )
}
