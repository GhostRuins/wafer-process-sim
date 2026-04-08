import { scaleSequential } from 'd3-scale'
import {
  interpolateRdBu,
  interpolateRdYlGn,
  interpolateViridis,
} from 'd3-scale-chromatic'

export type ColorScaleName = 'viridis' | 'coolwarm' | 'rdylgn' | 'tealGreen'

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

/** Fab-style teal → green for passing-metric visualization. */
function interpolateTealGreen(t: number): string {
  const r = lerp(0, 22, t)
  const g = lerp(212, 163, t)
  const b = lerp(170, 74, t)
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`
}

const interpolators = {
  viridis: interpolateViridis,
  coolwarm: interpolateRdBu,
  rdylgn: interpolateRdYlGn,
  tealGreen: interpolateTealGreen,
} as const

export function createMetricColorScale(
  name: ColorScaleName,
  domain: [number, number],
): (value: number) => string {
  const [a, b] = domain
  if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) {
    return () => '#334155'
  }
  const low = Math.min(a, b)
  const high = Math.max(a, b)
  const seq = scaleSequential(interpolators[name]).domain([low, high])
  return (value: number) =>
    Number.isFinite(value) ? String(seq(value)) : '#334155'
}
