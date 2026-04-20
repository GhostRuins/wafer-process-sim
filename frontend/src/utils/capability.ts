import type { ProcessRun } from '../hooks/useWaferData'
import type {
  CapabilityParamKey,
  ProcessParameterSpec,
} from '../config/processSpecs'

export type CapabilityStatus = 'incapable' | 'marginal' | 'capable' | 'insufficient'

export interface CapabilityIndexResult {
  cp: number | null
  cpk: number | null
  pp: number | null
  ppk: number | null
  mean: number | null
  sigmaOverall: number | null
  sigmaWithin: number | null
  sampleCount: number
  belowLslCount: number
  aboveUslCount: number
  inSpecCount: number
  status: CapabilityStatus
}

export interface CapabilityRow {
  parameter: ProcessParameterSpec
  indices: CapabilityIndexResult
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null
  return values.reduce((s, v) => s + v, 0) / values.length
}

function sampleStdev(values: number[]): number | null {
  if (values.length < 2) return null
  const m = mean(values)
  if (m === null) return null
  const variance =
    values.reduce((s, v) => s + (v - m) ** 2, 0) / (values.length - 1)
  return Math.sqrt(variance)
}

function movingRangeSigma(values: number[]): number | null {
  if (values.length < 2) return null
  const ranges: number[] = []
  for (let i = 1; i < values.length; i += 1) {
    ranges.push(Math.abs(values[i] - values[i - 1]))
  }
  const mrBar = mean(ranges)
  if (mrBar === null) return null
  const d2 = 1.128
  return mrBar / d2
}

function safeCapability(numerator: number, denominator: number | null): number | null {
  if (denominator === null || denominator <= 0 || !Number.isFinite(denominator)) {
    return null
  }
  const out = numerator / denominator
  return Number.isFinite(out) ? out : null
}

function classifyStatus(cpk: number | null, ppk: number | null): CapabilityStatus {
  const anchor = cpk ?? ppk
  if (anchor === null) return 'insufficient'
  if (anchor < 1) return 'incapable'
  if (anchor < 1.33) return 'marginal'
  return 'capable'
}

export function calculateCapabilityIndices(
  values: number[],
  limits: { lsl: number; usl: number },
): CapabilityIndexResult {
  const sampleCount = values.length
  const m = mean(values)
  const sigmaOverall = sampleStdev(values)
  const sigmaWithin = movingRangeSigma(values)
  const belowLslCount = values.filter((v) => v < limits.lsl).length
  const aboveUslCount = values.filter((v) => v > limits.usl).length
  const inSpecCount = sampleCount - belowLslCount - aboveUslCount

  if (m === null || sampleCount < 2) {
    return {
      cp: null,
      cpk: null,
      pp: null,
      ppk: null,
      mean: m,
      sigmaOverall,
      sigmaWithin,
      sampleCount,
      belowLslCount,
      aboveUslCount,
      inSpecCount,
      status: 'insufficient',
    }
  }

  const specWidth = limits.usl - limits.lsl
  const cp = safeCapability(specWidth, sigmaWithin === null ? null : 6 * sigmaWithin)
  const pp = safeCapability(specWidth, sigmaOverall === null ? null : 6 * sigmaOverall)

  const cpu = safeCapability(limits.usl - m, sigmaWithin === null ? null : 3 * sigmaWithin)
  const cpl = safeCapability(m - limits.lsl, sigmaWithin === null ? null : 3 * sigmaWithin)
  const ppu = safeCapability(limits.usl - m, sigmaOverall === null ? null : 3 * sigmaOverall)
  const ppl = safeCapability(m - limits.lsl, sigmaOverall === null ? null : 3 * sigmaOverall)

  const cpk = cpu !== null && cpl !== null ? Math.min(cpu, cpl) : null
  const ppk = ppu !== null && ppl !== null ? Math.min(ppu, ppl) : null

  return {
    cp,
    cpk,
    pp,
    ppk,
    mean: m,
    sigmaOverall,
    sigmaWithin,
    sampleCount,
    belowLslCount,
    aboveUslCount,
    inSpecCount,
    status: classifyStatus(cpk, ppk),
  }
}

export function computeCapabilityRows(
  runs: ProcessRun[],
  specs: Record<CapabilityParamKey, ProcessParameterSpec>,
): CapabilityRow[] {
  const keys = Object.keys(specs) as CapabilityParamKey[]
  return keys.map((key) => {
    const values = runs
      .map((r) => r[key])
      .filter((v): v is number => Number.isFinite(v))
    const indices = calculateCapabilityIndices(values, specs[key].specLimits)
    return {
      parameter: specs[key],
      indices,
    }
  })
}
