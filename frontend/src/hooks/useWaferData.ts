import { useQuery, type UseQueryResult } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

import type { DieData } from '../components/WaferMap'
import { apiUrl } from '../config'

/** Default dashboard range aligned with generated parquet (Jan 2026 lot). */
export function defaultProcessRunDateRange(): {
  start: string
  end: string
} {
  return {
    start: new Date('2026-01-01').toISOString(),
    end: new Date('2026-02-28').toISOString(),
  }
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Request failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

/** Single process run / wafer summary — aligns with Agent 6-style `/api/process-runs`. */
export interface ProcessRun {
  run_id: string
  wafer_id: string
  lot_id?: string
  lot_position?: number
  tool_id: string
  /** 0–1 fraction */
  wafer_yield: number
  timestamp: string
  anomaly_injected?: boolean
  anomaly_type?: string | null
  temperature: number
  pressure: number
  gas_flow: number
  rf_power: number
  deposition_time: number
}

function normalizeRunsPayload(body: unknown): ProcessRun[] {
  let raw: unknown[] = []
  if (Array.isArray(body)) raw = body
  else if (body && typeof body === 'object' && 'runs' in body) {
    const r = (body as { runs: unknown }).runs
    if (Array.isArray(r)) raw = r
  } else if (body && typeof body === 'object' && 'data' in body) {
    const r = (body as { data: unknown }).data
    if (Array.isArray(r)) raw = r
  }
  return raw
    .filter((x): x is Record<string, unknown> => Boolean(x && typeof x === 'object'))
    .map((r) => ({
      run_id: String(r.run_id ?? ''),
      wafer_id: String(r.wafer_id ?? ''),
      lot_id: r.lot_id ? String(r.lot_id) : undefined,
      lot_position:
        r.lot_position === undefined ? undefined : Number(r.lot_position),
      tool_id: String(r.tool_id ?? ''),
      wafer_yield: Number(r.wafer_yield ?? 0),
      timestamp: String(r.timestamp ?? ''),
      anomaly_injected: Boolean(r.anomaly_injected),
      anomaly_type: r.anomaly_type ? String(r.anomaly_type) : null,
      temperature: Number(r.temperature ?? 0),
      pressure: Number(r.pressure ?? 0),
      gas_flow: Number(r.gas_flow ?? 0),
      rf_power: Number(r.rf_power ?? 0),
      deposition_time: Number(r.deposition_time ?? 0),
    }))
}

function extractDiesArray(body: unknown): Record<string, unknown>[] {
  let raw: unknown[] = []
  if (Array.isArray(body)) raw = body
  else if (body && typeof body === 'object') {
    const o = body as Record<string, unknown>
    if (Array.isArray(o.dies)) raw = o.dies
    else if (Array.isArray(o.data)) raw = o.data
    else if (Array.isArray(o.items)) raw = o.items
  }
  return raw.filter((x): x is Record<string, unknown> => x !== null && typeof x === 'object')
}

function normalizeDieRow(row: Record<string, unknown>): DieData {
  const x = Number(row.x_pos ?? row.x ?? 0)
  const y = Number(row.y_pos ?? row.y ?? 0)
  const film_thickness = Number(row.film_thickness ?? row.film_thickness_nm ?? 0)
  const defect_density = Number(row.defect_density ?? 0)
  const die_yield = Number(row.die_yield ?? 0)
  const pf = row.pass_fail
  const pass_fail =
    typeof pf === 'boolean' ? pf : typeof pf === 'number' ? pf !== 0 : die_yield > 0.85
  return { x, y, film_thickness, defect_density, die_yield, pass_fail }
}

function normalizeDiesPayload(body: unknown): DieData[] {
  return extractDiesArray(body).map(normalizeDieRow)
}

export async function fetchProcessRuns(
  startIso: string,
  endIso: string,
): Promise<ProcessRun[]> {
  const q = new URLSearchParams({ start: startIso, end: endIso })
  const res = await fetch(apiUrl(`/api/process-runs?${q}`))
  const body = await parseJson<unknown>(res)
  return normalizeRunsPayload(body)
}

export async function fetchRunDies(runId: string): Promise<DieData[]> {
  const res = await fetch(apiUrl(`/api/runs/${encodeURIComponent(runId)}/dies`))
  const body = await parseJson<unknown>(res)
  return normalizeDiesPayload(body)
}

export interface UseWaferDataResult {
  runsQuery: UseQueryResult<ProcessRun[], Error>
  diesQuery: UseQueryResult<DieData[], Error>
  runs: ProcessRun[]
  sortedRuns: ProcessRun[]
  selectedRun: ProcessRun | undefined
  runIndex: number
  setRunIndex: (i: number | ((prev: number) => number)) => void
  goPrevRun: () => void
  goNextRun: () => void
}

export function useWaferData(
  startIso: string,
  endIso: string,
): UseWaferDataResult {
  const runsQuery = useQuery({
    queryKey: ['process-runs', startIso, endIso],
    queryFn: () => fetchProcessRuns(startIso, endIso),
  })

  const sortedRuns = useMemo(() => {
    const r = [...(runsQuery.data ?? [])]
    r.sort((a, b) => a.timestamp.localeCompare(b.timestamp))
    return r
  }, [runsQuery.data])

  const [runIndex, setRunIndex] = useState(0)

  useEffect(() => {
    if (sortedRuns.length === 0) return
    setRunIndex((i) => Math.min(i, sortedRuns.length - 1))
  }, [sortedRuns.length])

  const selectedRun = sortedRuns[runIndex]

  const diesQuery = useQuery({
    queryKey: ['run-dies', selectedRun?.run_id],
    queryFn: () => fetchRunDies(selectedRun!.run_id),
    enabled: Boolean(selectedRun?.run_id),
  })

  const goPrevRun = () =>
    setRunIndex((i) => Math.max(0, i - 1))

  const goNextRun = () =>
    setRunIndex((i) => Math.min(sortedRuns.length - 1, i + 1))

  return {
    runsQuery,
    diesQuery,
    runs: runsQuery.data ?? [],
    sortedRuns,
    selectedRun,
    runIndex,
    setRunIndex,
    goPrevRun,
    goNextRun,
  }
}

export function computeKpis(runs: ProcessRun[], dayIso: string) {
  const dayPrefix = dayIso.slice(0, 10)
  const todayRuns = runs.filter((r) => r.timestamp.startsWith(dayPrefix))
  const avgYieldToday =
    todayRuns.length > 0
      ? todayRuns.reduce((s, r) => s + r.wafer_yield, 0) / todayRuns.length
      : null

  const byTool = new Map<string, { sum: number; n: number }>()
  for (const r of runs) {
    const cur = byTool.get(r.tool_id) ?? { sum: 0, n: 0 }
    cur.sum += r.wafer_yield
    cur.n += 1
    byTool.set(r.tool_id, cur)
  }
  let bestTool: string | null = null
  let bestAvg = -1
  for (const [tool, { sum, n }] of byTool) {
    const a = sum / n
    if (a > bestAvg) {
      bestAvg = a
      bestTool = tool
    }
  }

  return {
    avgYieldToday,
    totalRuns: runs.length,
    bestTool: bestTool ?? '—',
    bestToolAvgYield: bestTool ? bestAvg : null,
  }
}

export type QuadrantId = 'NW' | 'NE' | 'SW' | 'SE'

export function worstDieClusterSummary(dies: DieData[]): {
  label: string
  quadrant: QuadrantId
  meanYield: number
} | null {
  if (dies.length === 0) return null

  const xs = dies.map((d) => d.x)
  const ys = dies.map((d) => d.y)
  const midX = (Math.min(...xs) + Math.max(...xs)) / 2
  const midY = (Math.min(...ys) + Math.max(...ys)) / 2

  const quads: Record<QuadrantId, number[]> = {
    NW: [],
    NE: [],
    SW: [],
    SE: [],
  }

  for (const d of dies) {
    const q: QuadrantId =
      d.x < midX ? (d.y < midY ? 'NW' : 'SW') : d.y < midY ? 'NE' : 'SE'
    quads[q].push(d.die_yield)
  }

  let worst: QuadrantId | null = null
  let worstMean = Infinity
  for (const k of Object.keys(quads) as QuadrantId[]) {
    const vals = quads[k]
    if (vals.length === 0) continue
    const m = vals.reduce((a, b) => a + b, 0) / vals.length
    if (m < worstMean) {
      worstMean = m
      worst = k
    }
  }

  if (!worst) return null
  return {
    quadrant: worst,
    meanYield: worstMean,
    label: `${worst} cluster μ=${(worstMean * 100).toFixed(1)}%`,
  }
}

export function runsToCsv(runs: ProcessRun[]): string {
  const cols = [
    'run_id',
    'wafer_id',
    'tool_id',
    'timestamp',
    'wafer_yield',
    'anomaly_injected',
    'temperature',
    'pressure',
    'gas_flow',
    'rf_power',
    'deposition_time',
  ] as const
  const esc = (v: unknown) => {
    const s =
      v === null || v === undefined
        ? ''
        : typeof v === 'boolean'
          ? v
            ? 'true'
            : 'false'
          : String(v)
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`
    return s
  }
  const lines = [cols.join(',')]
  for (const r of runs) {
    lines.push(
      cols
        .map((c) =>
          esc(
            c === 'anomaly_injected' ? Boolean(r.anomaly_injected) : r[c],
          ),
        )
        .join(','),
    )
  }
  return lines.join('\n')
}
