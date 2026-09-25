/**
 * What a page's hardware panels show by default: this machine, or a peer whose
 * hardware the dashboard mirrors over its own `/ws` stream (`--remote`).
 *
 * This is the hardware half of the binding model, made persistent — the page
 * source but for machine rather than model. `pageSource.ts` answers "which
 * engine's numbers"; this answers "which machine's numbers", which a mirrored
 * fleet view needs: a page that reads dgx1's GPU and thermals while everything
 * else stays local.
 *
 * **Absent means this machine.** A page with no hardware source reads the
 * host's own hardware exactly as every page did before mirrored peers existed,
 * which is what keeps the schema migration additive. Only the remote case is
 * worth storing; "local" is what absence already says.
 */

import { isRecord } from './json'
import type { MetricsSnapshot, RemoteHostSnapshot } from '@/types/metrics'

/**
 * The operator's choice, as stored on the page. Only the remote form is ever
 * stored — absent (`undefined`) is local, and a stored `{ kind: 'local' }`
 * would be a second way of saying nothing.
 *
 * `url` is the mirrored peer's address as the backend lists it in the
 * snapshot's `remote[]` — the same identity a peer panel binding would use.
 */
export interface HardwareSource {
  readonly kind: 'remote'
  readonly url: string
}

/**
 * Reads a persisted hardware source. Absent or malformed becomes `undefined`
 * — local. As with a page's engine source, a source that cannot be read is
 * not kept as a state of its own: the page falls back, silently in the
 * reading layer, visibly in the panels that then find no reason to explain.
 */
export function readHardwareSource(raw: unknown): HardwareSource | undefined {
  if (!isRecord(raw)) return undefined
  if (raw.kind === 'remote' && typeof raw.url === 'string' && raw.url.length > 0) {
    return { kind: 'remote', url: raw.url }
  }
  return undefined
}

/** The option value of local hardware, which is not stored. */
const LOCAL_CHOICE = 'local'

/** One machine the page could show hardware from, as the control offers it. */
export interface HardwareChoice {
  /** Round-trips through `hardwareFromChoice`; safe as an option value. */
  value: string
  label: string
  /** The peer is not mirrored on this host — a source left dangling. */
  absent?: boolean
}

/** Everything the control needs: what to offer, and what is chosen now. */
export interface HardwareControlModel {
  value: string
  choices: HardwareChoice[]
}

/** A mirrored peer as the control knows it. */
export interface MirroredPeer {
  url: string
  label: string
}

/**
 * The machines a page can be pointed at, with the one it holds selected.
 *
 * Same prohibition as `pageSourceChoices`: a source naming a peer that is not
 * mirrored here keeps its own option, marked absent and selected, rather than
 * being quietly shown as this machine's numbers. A peer that is listed but
 * currently down is still offered — reconnect is automatic, so choosing the
 * machine behind the outage is a legitimate thing for a page to do.
 */
export function hardwareChoices(
  source: HardwareSource | undefined,
  peers: readonly MirroredPeer[],
): HardwareControlModel {
  const choices: HardwareChoice[] = [
    { value: LOCAL_CHOICE, label: 'This machine' },
    ...peers.map((peer) => ({ value: choiceOf(peer.url), label: peer.label })),
  ]

  const value = source === undefined ? LOCAL_CHOICE : choiceOf(source.url)

  const missing =
    source !== undefined && !peers.some((peer) => peer.url === source.url)
      ? { value, label: `${source.url} (not mirrored here)`, absent: true }
      : null

  return { value, choices: missing ? [...choices, missing] : choices }
}

/**
 * The source a chosen option means. Null is local — the caller removes the
 * field rather than storing a sentinel. Anything not in this module's own
 * grammar reads as local too; the values come from `hardwareChoices`.
 */
export function hardwareFromChoice(value: string): HardwareSource | null {
  if (value === LOCAL_CHOICE) return null

  const separator = value.indexOf(':')
  if (value.slice(0, separator) === 'remote') {
    const url = value.slice(separator + 1)
    if (url.length > 0) return { kind: 'remote', url }
  }

  return null
}

/** The one place a peer URL becomes an option value. */
function choiceOf(url: string): string {
  return `remote:${url}`
}

/**
 * A hardware source resolved against the live snapshot. Same rule as every
 * other resolution on this dashboard: a source that cannot be honored names
 * what is wrong rather than substituting — a peer that is not mirrored here is
 * shown as not mirrored, and one that is down is shown as down, never as this
 * machine's numbers under a title that implied another box.
 */
export type HardwareResolution =
  /** No source: this machine's hardware. */
  | { status: 'local' }
  /** No snapshot has arrived yet, so neither machine can be confirmed. */
  | { status: 'waiting' }
  /** The peer is mirrored and connected. `host` carries its live snapshot. */
  | { status: 'remote'; host: RemoteHostSnapshot }
  /** The page points at a peer this backend mirrors no connection to. */
  | { status: 'not-mirrored'; url: string }
  /** The peer is mirrored but not currently reachable. */
  | { status: 'down'; url: string }

/** Resolve `source` against `snapshot.remote`. A snapshot is required: while
 *  none has arrived nothing is known about either machine. */
export function resolveHardware(
  source: HardwareSource | undefined,
  snapshot: Pick<MetricsSnapshot, 'remote'> | null,
): HardwareResolution {
  if (!source) return { status: 'local' }
  if (!snapshot) return { status: 'waiting' }

  const host = (snapshot.remote ?? []).find((entry) => entry.url === source.url)
  if (!host) return { status: 'not-mirrored', url: source.url }
  if (!host.connected || !host.data) return { status: 'down', url: host.url }
  return { status: 'remote', host }
}
