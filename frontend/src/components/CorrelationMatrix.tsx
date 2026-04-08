import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { ProcessRun } from '../hooks/useWaferData'
import {
  chartGridProps,
  chartTickProps,
  chartTooltipContentStyle,
} from '../styles/chartTheme'

import styles from './CorrelationMatrix.module.css'

const LABELS = [
  'temperature',
  'pressure',
  'gas_flow',
  'rf_power',
  'deposition_time',
  'wafer_yield',
] as const

const SHORT: Record<(typeof LABELS)[number], string> = {
  temperature: 'TEMP',
  pressure: 'PRS',
  gas_flow: 'GAS',
  rf_power: 'RF',
  deposition_time: 'TIME',
  wafer_yield: 'YLD',
}

type VarKey = (typeof LABELS)[number]

function pearson(xs: number[], ys: number[]): number {
  const n = xs.length
  if (n < 2 || n !== ys.length) return 0
  const mx = xs.reduce((a, b) => a + b, 0) / n
  const my = ys.reduce((a, b) => a + b, 0) / n
  let num = 0
  let dx = 0
  let dy = 0
  for (let i = 0; i < n; i++) {
    const vx = xs[i]! - mx
    const vy = ys[i]! - my
    num += vx * vy
    dx += vx * vx
    dy += vy * vy
  }
  const den = Math.sqrt(dx * dy)
  return den === 0 ? 0 : num / den
}

function mix(a: number, b: number, t: number) {
  return Math.round(a + (b - a) * t)
}

/** Diverging: negative → red tint, positive → teal tint, ~0 → bg-elevated */
function cellColor(r: number): string {
  const base: [number, number, number] = [26, 34, 53]
  if (Math.abs(r) < 0.035) {
    return `rgb(${base[0]}, ${base[1]}, ${base[2]})`
  }
  const t = Math.min(1, Math.abs(r)) * 0.78
  if (r < 0) {
    return `rgb(${mix(base[0], 239, t)}, ${mix(base[1], 68, t)}, ${mix(base[2], 68, t)})`
  }
  return `rgb(${mix(base[0], 0, t)}, ${mix(base[1], 212, t)}, ${mix(base[2], 170, t)})`
}

function textFillForCell(r: number): string {
  if (Math.abs(r) < 0.035) return 'var(--text-muted)'
  return Math.abs(r) > 0.45 ? '#f1f5f9' : '#0a0e1a'
}

export function CorrelationMatrix({
  runs,
  loading,
}: {
  runs: ProcessRun[]
  loading?: boolean
}) {
  const [open, setOpen] = useState<{ i: number; j: number } | null>(null)
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null)

  const { matrix, series } = useMemo(() => {
    const cols: VarKey[] = [...LABELS]
    const series: Record<VarKey, number[]> = {
      temperature: [],
      pressure: [],
      gas_flow: [],
      rf_power: [],
      deposition_time: [],
      wafer_yield: [],
    }
    for (const r of runs) {
      series.temperature.push(r.temperature)
      series.pressure.push(r.pressure)
      series.gas_flow.push(r.gas_flow)
      series.rf_power.push(r.rf_power)
      series.deposition_time.push(r.deposition_time)
      series.wafer_yield.push(r.wafer_yield)
    }
    const n = cols.length
    const matrix: number[][] = Array.from({ length: n }, () =>
      Array.from({ length: n }, () => 0),
    )
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        matrix[i]![j] = pearson(series[cols[i]!]!, series[cols[j]!]!)
      }
    }
    return { matrix, series }
  }, [runs])

  const scatterData = useMemo(() => {
    if (!open) return []
    const xi = LABELS[open.i]!
    const yi = LABELS[open.j]!
    const xs = series[xi]
    const ys = series[yi]
    if (!xs || !ys) return []
    return xs.map((x, k) => ({ x, y: ys[k]!, i: k }))
  }, [open, series])

  if (loading) {
    return (
      <div className={styles.root}>
        <div className={`${styles.skeleton} skeleton`} />
      </div>
    )
  }

  const n = LABELS.length
  const pad = 20
  const inner = 100
  const cell = inner / n
  const vb = pad + inner + 4

  const dim = (i: number, j: number) => {
    if (!hover) return 1
    if (hover.i === i || hover.j === j) return 1
    return 0.38
  }

  return (
    <div className={styles.root}>
      <div className={styles.head}>
        <h3 className={styles.title}>Correlation matrix</h3>
        <p className={styles.sub}>Pearson r · click cell · scatter modal</p>
      </div>

      {runs.length < 3 ? (
        <p className={styles.empty}>Need at least 3 runs for correlations.</p>
      ) : (
        <div className={styles.chartWrap}>
          <svg
            viewBox={`0 0 ${vb} ${vb}`}
            className={styles.svg}
            role="img"
          >
            {LABELS.map((_, j) => (
              <text
                key={`tc-${j}`}
                x={pad + j * cell + cell / 2}
                y={12}
                textAnchor="middle"
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 6.5,
                  fill:
                    hover?.j === j ? 'var(--accent-teal)' : 'var(--text-muted)',
                  fontWeight: hover?.j === j ? 700 : 500,
                }}
              >
                {SHORT[LABELS[j]!]}
              </text>
            ))}
            {LABELS.map((_, i) => (
              <text
                key={`lr-${i}`}
                x={10}
                y={pad + i * cell + cell / 2 + 2}
                textAnchor="middle"
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 6.5,
                  fill:
                    hover?.i === i ? 'var(--accent-teal)' : 'var(--text-muted)',
                  fontWeight: hover?.i === i ? 700 : 500,
                }}
              >
                {SHORT[LABELS[i]!]}
              </text>
            ))}
            {matrix.flatMap((row, i) =>
              row.map((r, j) => {
                const x = pad + j * cell
                const y = pad + i * cell
                return (
                  <g key={`${i}-${j}`} opacity={dim(i, j)}>
                    <rect
                      x={x}
                      y={y}
                      width={cell - 0.35}
                      height={cell - 0.35}
                      rx={1}
                      fill={cellColor(r)}
                      stroke="var(--border)"
                      strokeWidth={0.12}
                      className="cursor-pointer interactive"
                      onClick={() => setOpen({ i, j })}
                      onMouseEnter={() => setHover({ i, j })}
                      onMouseLeave={() => setHover(null)}
                    />
                    <text
                      x={x + cell / 2}
                      y={y + cell / 2 + 2}
                      textAnchor="middle"
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 5.8,
                        fill: textFillForCell(r),
                        pointerEvents: 'none',
                      }}
                    >
                      {r.toFixed(2)}
                    </text>
                  </g>
                )
              }),
            )}
          </svg>
        </div>
      )}

      {open && (
        <div
          className={styles.modal}
          role="dialog"
          aria-modal
          onClick={() => setOpen(null)}
        >
            <div
              className={styles.modalInner}
              onClick={(e) => e.stopPropagation()}
            >
              <div className={styles.modalHead}>
                <h4 className={styles.modalTitle}>
                  {LABELS[open.i]} vs {LABELS[open.j]}
                </h4>
                <button
                  type="button"
                  className={styles.closeBtn}
                  onClick={() => setOpen(null)}
                >
                  Close
                </button>
              </div>
              <div className={styles.modalChart}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                    <CartesianGrid {...chartGridProps} />
                    <XAxis
                      type="number"
                      dataKey="x"
                      name={LABELS[open.i]}
                      stroke="var(--border)"
                      tick={chartTickProps}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name={LABELS[open.j]}
                      stroke="var(--border)"
                      tick={chartTickProps}
                    />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      contentStyle={chartTooltipContentStyle}
                    />
                    <Scatter data={scatterData} fill="var(--accent-teal)" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
        </div>
      )}
    </div>
  )
}
