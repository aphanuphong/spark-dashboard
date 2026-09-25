import { IoPanel } from './IoPanel'
import { HardwarePanelNotice } from './PanelNotice'
import { hardwareDevice, useHardwarePanelSeries } from './useHardwarePanel'
import type { PanelContentProps } from '../panelRegistry'

/** Network receive/transmit rates and their trend. Host-wide, so it follows
 *  the page's hardware source. */
export function NetworkIoPanel({ panel }: PanelContentProps) {
  const rx = useHardwarePanelSeries('networkRx', panel.window)
  const tx = useHardwarePanelSeries('networkTx', panel.window)
  const resolution = rx.resolution
  if (resolution.status === 'peer-not-mirrored' || resolution.status === 'peer-down') {
    return <HardwarePanelNotice resolution={resolution} />
  }
  const snapshot = resolution.status === 'resolved' ? resolution.snapshot : null

  return (
    <IoPanel
      device={hardwareDevice(snapshot?.network.name, resolution)}
      inbound={{
        tag: 'RX',
        label: 'RX',
        color: '#3B82F6',
        rate: snapshot ? snapshot.network.rx_bytes_per_sec : null,
        data: rx.data,
      }}
      outbound={{
        tag: 'TX',
        label: 'TX',
        color: '#A855F7',
        rate: snapshot ? snapshot.network.tx_bytes_per_sec : null,
        data: tx.data,
      }}
    />
  )
}
