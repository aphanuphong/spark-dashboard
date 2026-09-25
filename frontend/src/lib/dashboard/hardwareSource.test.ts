import { describe, expect, it } from 'vitest'
import {
  hardwareChoices,
  hardwareFromChoice,
  readHardwareSource,
  resolveHardware,
} from './hardwareSource'
import type { MetricsSnapshot, RemoteHostSnapshot } from '@/types/metrics'

const DGX = 'http://dgx1.rt-ctrl.com:3000'

function peer(
  url: string,
  label: string,
  connected = true,
  data: MetricsSnapshot | null = null,
): RemoteHostSnapshot {
  return { url, label, connected, data }
}

describe('readHardwareSource', () => {
  it('reads the one stored kind', () => {
    expect(readHardwareSource({ kind: 'remote', url: DGX })).toEqual({ kind: 'remote', url: DGX })
  })

  it('reads everything else as this machine', () => {
    // Absent is local everywhere else in the document, and a malformed field
    // must not create a state of its own: the page falls back quietly.
    expect(readHardwareSource(undefined)).toBeUndefined()
    expect(readHardwareSource(null)).toBeUndefined()
    expect(readHardwareSource('remote')).toBeUndefined()
    expect(readHardwareSource({ kind: 'remote' })).toBeUndefined()
    expect(readHardwareSource({ kind: 'remote', url: '' })).toBeUndefined()
    expect(readHardwareSource({ kind: 'local' })).toBeUndefined()
  })
})

describe('hardwareChoices', () => {
  const peers = [peer(DGX, 'dgx1'), peer('http://maven:3000', 'maven')]

  it('offers this machine and every mirrored peer', () => {
    const { value, choices } = hardwareChoices(undefined, peers)

    expect(value).toBe('local')
    expect(choices.map((choice) => choice.label)).toEqual(['This machine', 'dgx1', 'maven'])
  })

  it('selects the stored source without hiding anything', () => {
    expect(hardwareChoices({ kind: 'remote', url: DGX }, peers).value).toBe(`remote:${DGX}`)
    // A peer that is merely down is still a machine a page may point at.
    expect(
      hardwareChoices(
        { kind: 'remote', url: DGX },
        peers.map((entry) => (entry.url === DGX ? { ...entry, connected: false } : entry)),
      ).choices,
    ).toHaveLength(3)
  })

  it('keeps a configured peer that is not mirrored here, marked absent', () => {
    // The prohibition on silent substitution starts at the control: an option
    // that is not here keeps its own name, selected and marked, rather than
    // being quietly shown as this machine's numbers.
    const { value, choices } = hardwareChoices({ kind: 'remote', url: 'http://gone:3000' }, peers)

    expect(value).toBe('remote:http://gone:3000')
    expect(choices.at(-1)).toEqual({
      value: 'remote:http://gone:3000',
      label: 'http://gone:3000 (not mirrored here)',
      absent: true,
    })
  })

  it('round-trips every choice it offers', () => {
    const { choices } = hardwareChoices(undefined, peers)

    expect(hardwareFromChoice(choices[0].value)).toBeNull()
    expect(hardwareFromChoice(choices[1].value)).toEqual({ kind: 'remote', url: DGX })
  })
})

describe('hardwareFromChoice', () => {
  it('reads local and anything outside the grammar as this machine', () => {
    // Null is the signal to remove the field, not to store a sentinel.
    expect(hardwareFromChoice('local')).toBeNull()
    expect(hardwareFromChoice('remote:')).toBeNull()
    expect(hardwareFromChoice('all')).toBeNull()
    expect(hardwareFromChoice('')).toBeNull()
  })

  it('keeps a URL whose own value contains colons', () => {
    // The whole point of the `remote:` prefix: the rest is taken verbatim.
    expect(hardwareFromChoice(`remote:${DGX}`)).toEqual({ kind: 'remote', url: DGX })
  })
})

describe('resolveHardware', () => {
  const snapshot = (remote?: RemoteHostSnapshot[]): Pick<MetricsSnapshot, 'remote'> => ({ remote })

  it('resolves an absent source as this machine, with no snapshot required', () => {
    expect(resolveHardware(undefined, null)).toEqual({ status: 'local' })
    expect(resolveHardware(undefined, snapshot())).toEqual({ status: 'local' })
  })

  it('waits while no snapshot has arrived', () => {
    expect(resolveHardware({ kind: 'remote', url: DGX }, null)).toEqual({ status: 'waiting' })
  })

  it('resolves a connected peer to its forwarded snapshot', () => {
    const host = peer(DGX, 'dgx1', true, {} as MetricsSnapshot)

    expect(resolveHardware({ kind: 'remote', url: DGX }, snapshot([host]))).toEqual({
      status: 'remote',
      host,
    })
  })

  it('names a peer that is not mirrored rather than substituting', () => {
    expect(resolveHardware({ kind: 'remote', url: DGX }, snapshot([]))).toEqual({
      status: 'not-mirrored',
      url: DGX,
    })
    expect(resolveHardware({ kind: 'remote', url: DGX }, snapshot())).toEqual({
      status: 'not-mirrored',
      url: DGX,
    })
  })

  it('names a mirrored peer that is currently down', () => {
    expect(
      resolveHardware({ kind: 'remote', url: DGX }, snapshot([peer(DGX, 'dgx1', false)])),
    ).toEqual({ status: 'down', url: DGX })

    // Listed as connected but forwarding nothing is still "down" — there is no
    // data to substitute, and a panel that invented some would be lying.
    expect(
      resolveHardware({ kind: 'remote', url: DGX }, snapshot([peer(DGX, 'dgx1', true, null)])),
    ).toEqual({ status: 'down', url: DGX })
  })
})
