import { MetricTile } from '@/components/engines/EnginePanelPrimitives'
import { useLatestSnapshot } from '@/hooks/useMetricsStore'
import { formatGiB, formatPercent, formatWatts, formatTemp } from '@/lib/format'
import type { RemoteHostSnapshot } from '@/types/metrics'
import { PanelNotice } from './PanelNotice'
import type { PanelContentProps } from '../panelRegistry'

/**
 * Peer hosts whose hardware this dashboard mirrors over their own `/ws`
 * stream (the `--remote` flag on the backend). One card per peer, each
 * showing that machine's headline hardware numbers side by side.
 *
 * The live values are read from the peer snapshot data verbatim — same wire
 * format, same field names — so a peer's GPU row reads identically to the
 * local GPU panel. No series history: the history store keys series to local
 * snapshots, so charts stay a local-host concern and this panel stays an
 * at-a-glance fleet readout.
 */
export function RemoteHostsPanel(_props: PanelContentProps) {
  const snapshot = useLatestSnapshot()
  const hosts = snapshot?.remote

  if (!snapshot) return <PanelNotice>Waiting for metrics</PanelNotice>
  if (!hosts || hosts.length === 0) {
    return (
      <PanelNotice>
        No remote hosts configured. Start with `--remote` /
        `SPARK_DASHBOARD_REMOTE` to mirror another dashboard.
      </PanelNotice>
    )
  }

  return (
    <div className="h-full overflow-y-auto p-2 flex flex-col gap-2">
      {hosts.map((host) => (
        <HostCard key={host.url} host={host} />
      ))}
    </div>
  )
}

function HostCard({ host }: { host: RemoteHostSnapshot }) {
  const gpu = host.data?.gpu
  return (
    <div className="rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-2">
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="text-xs font-medium text-zinc-200 truncate">{host.label}</span>
        <span
          className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider ${
            host.connected
              ? 'bg-[#76B900]/15 text-[#76B900]'
              : 'bg-red-500/15 text-red-400'
          }`}
        >
          {host.connected ? 'Live' : 'Down'}
        </span>
      </div>
      {host.connected && gpu ? (
        <div className="grid grid-cols-2 min-[420px]:grid-cols-4 gap-x-3 gap-y-1">
          <MetricTile
            label="GPU"
            value={formatPercent(gpu.utilization_percent)}
            subline={gpu.name ?? undefined}
          />
          <MetricTile
            label="Temp"
            value={formatTemp(gpu.temperature_celsius)}
            warn={(gpu.temperature_celsius ?? 0) > 85}
          />
          <MetricTile label="Power" value={formatWatts(host.data!.gpu.power_watts ?? 0)} />
          <MetricTile
            label="Memory"
            value={
              gpu.memory_total_bytes
                ? `${formatGiB(gpu.memory_used_bytes ?? 0)}/${formatGiB(gpu.memory_total_bytes)}`
                : formatGiB(host.data!.memory.used_bytes)
            }
          />
        </div>
      ) : (
        <p className="text-[11px] text-zinc-500">
          {host.connected
            ? 'Connected, waiting for the first snapshot.'
            : `No data from ${host.url} — retrying automatically.`}
        </p>
      )}
    </div>
  )
}
