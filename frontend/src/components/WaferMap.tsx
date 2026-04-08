import { type ReactElement, useCallback, useId, useMemo, useState } from 'react'

import {
  type ColorScaleName,
  createMetricColorScale,
} from '../utils/colorScales'

import styles from './WaferMap.module.css'

export type WaferMetric = 'film_thickness' | 'defect_density' | 'die_yield'

export interface DieData {
  x: number
  y: number
  pass_fail: boolean
  film_thickness: number
  defect_density: number
  die_yield: number
  tool_id?: string
  run_id?: string
}

function normalizeCoord(v: number, min: number, max: number): number {
  // If API sends [0,1], convert to [-1,1].
  if (min >= 0 && max <= 1.000001) return (v - 0.5) * 2
  // If already [-1,1]-ish, keep as-is.
  if (min >= -1.1 && max <= 1.1) return v
  // Fallback for index-like coordinates.
  if (max > min) return ((v - min) / (max - min)) * 2 - 1
  return 0
}

function metricValue(d: DieData, m: WaferMetric): number {
  switch (m) {
    case 'film_thickness':
      return d.film_thickness
    case 'defect_density':
      return d.defect_density
    case 'die_yield':
      return d.die_yield
    default:
      return 0
  }
}

const METRIC_LABEL: Record<WaferMetric, string> = {
  film_thickness: 'Thickness (nm)',
  defect_density: 'Defect density',
  die_yield: 'Die yield',
}

export interface WaferMapProps {
  wafer_id: string
  metric: WaferMetric
  data: DieData[]
  colorScale?: ColorScaleName
  showExclusionZone?: boolean
  onDieClick?: (die: DieData) => void
}

const R = 180
const CX = 200
const CY = 200

export function WaferMap({
  wafer_id,
  metric,
  data,
  colorScale = 'tealGreen',
  showExclusionZone = false,
  onDieClick,
}: WaferMapProps) {
  const clipId = useId()
  const hatchId = useId()
  const [hover, setHover] = useState<DieData | null>(null)
  const [hoverKey, setHoverKey] = useState<string | null>(null)
  const [tipPos, setTipPos] = useState<{ x: number; y: number }>({
    x: 0,
    y: 0,
  })

  const { vals, normalizedDies, dieSizePx } = useMemo(() => {
    const vals = data.map((d) => metricValue(d, metric))
    const xs = data.map((d) => d.x)
    const ys = data.map((d) => d.y)
    const minX = xs.length ? Math.min(...xs) : 0
    const maxX = xs.length ? Math.max(...xs) : 1
    const minY = ys.length ? Math.min(...ys) : 0
    const maxY = ys.length ? Math.max(...ys) : 1
    const normalizedDies = data.map((d, idx) => ({
      die: d,
      key: `${d.x},${d.y},${idx}`,
      xNorm: normalizeCoord(d.x, minX, maxX),
      yNorm: normalizeCoord(d.y, minY, maxY),
    }))
    const inferredGrid = Math.max(1, Math.round(Math.sqrt(data.length || 1)))
    const dieSizePx = (2 * R) / inferredGrid
    return { vals, normalizedDies, dieSizePx }
  }, [data, metric])

  const { min, max } = useMemo(() => {
    if (vals.length === 0) return { min: 0, max: 1 }
    return { min: Math.min(...vals), max: Math.max(...vals) }
  }, [vals])

  const colorFor = useMemo(
    () => createMetricColorScale(colorScale, [min, max]),
    [colorScale, min, max],
  )

  const onSvgPointer = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (!hover) return
    setTipPos({ x: e.clientX + 14, y: e.clientY + 10 })
  }, [hover])

  const cells = useMemo(() => {
    const out: ReactElement[] = []
    for (const item of normalizedDies) {
      const die = item.die
      const v = metricValue(die, metric)
      const fill = colorFor(v)
      const key = item.key
      const dieCx = CX + item.xNorm * R
      const dieCy = CY + item.yNorm * R
      const left = dieCx - dieSizePx / 2
      const top = dieCy - dieSizePx / 2
      const sc = hoverKey === key ? 1.4 : 1
      out.push(
        <g
          key={key}
          transform={`translate(${dieCx},${dieCy}) scale(${sc}) translate(${-dieCx},${-dieCy})`}
          style={{ transition: 'transform 150ms ease' }}
        >
          <rect
            x={left}
            y={top}
            width={dieSizePx}
            height={dieSizePx}
            fill={fill}
            stroke="var(--bg-primary)"
            strokeWidth={0.5}
            className="cursor-pointer interactive"
            onPointerEnter={(e) => {
              setHover(die)
              setHoverKey(key)
              setTipPos({ x: e.clientX + 14, y: e.clientY + 10 })
            }}
            onPointerLeave={() => {
              setHover(null)
              setHoverKey(null)
            }}
            onPointerMove={(e) => {
              setTipPos({ x: e.clientX + 14, y: e.clientY + 10 })
            }}
            onClick={() => onDieClick?.(die)}
          />
          {!die.pass_fail && (
            <rect
              x={left}
              y={top}
              width={dieSizePx}
              height={dieSizePx}
              fill={`url(#${hatchId})`}
              pointerEvents="none"
            />
          )}
        </g>,
      )
    }
    return out
  }, [
    normalizedDies,
    dieSizePx,
    colorFor,
    metric,
    onDieClick,
    hatchId,
    hoverKey,
  ])

  const mean =
    vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0

  const legendCss = useMemo(() => {
    const n = 24
    const parts: string[] = []
    for (let k = 0; k <= n; k++) {
      const t = k / n
      const v = min + t * (max - min)
      parts.push(`${colorFor(v)} ${(t * 100).toFixed(1)}%`)
    }
    return `linear-gradient(to right, ${parts.join(', ')})`
  }, [min, max, colorFor])

  return (
    <div className={styles.root}>
      <p
        className="font-mono text-[10px] text-[var(--text-muted)]"
        style={{ fontFamily: 'var(--font-mono)' }}
      >
        {wafer_id}
      </p>

      <div className={styles.mapRow}>
        <div className={styles.svgWrap}>
          <svg
            viewBox="0 0 400 400"
            className={styles.svg}
            role="img"
            aria-label={`Wafer map for ${wafer_id}`}
            onPointerMove={onSvgPointer}
          >
            <defs>
              <clipPath id={clipId}>
                <circle cx={CX} cy={CY} r={R} />
              </clipPath>
              <pattern
                id={hatchId}
                patternUnits="userSpaceOnUse"
                width="6"
                height="6"
              >
                <path
                  d="M-1,7 l8,-8 M0,7 l8,-8 M1,7 l8,-8"
                  stroke="var(--accent-red)"
                  strokeWidth="1.2"
                  opacity={0.9}
                />
              </pattern>
            </defs>

            <circle
              cx={CX}
              cy={CY}
              r={R}
              fill="var(--bg-primary)"
              stroke="var(--border)"
              strokeWidth={1}
            />
            <g clipPath={`url(#${clipId})`}>{cells}</g>
            {showExclusionZone && (
              <circle
                cx={CX}
                cy={CY}
                r={R * 0.85}
                fill="none"
                stroke="var(--text-muted)"
                strokeWidth={0.5}
                strokeDasharray="4 3"
                opacity={0.85}
              />
            )}
          </svg>
        </div>

        <div className={styles.legend}>
          <div
            className={styles.legendTrack}
            style={{ background: legendCss }}
          />
          <p className={styles.legendMeta}>
            min {min.toFixed(3)} · max {max.toFixed(3)} · μ {mean.toFixed(3)}
          </p>
        </div>
      </div>

      {hover && (
        <div
          className={styles.tooltip}
          style={{
            left: tipPos.x,
            top: tipPos.y,
            transform: 'translate(0, 0)',
          }}
        >
          <div className={styles.tooltipRow}>
            <span style={{ color: 'var(--text-muted)' }}>Die</span>
            <span>
              ({hover.x}, {hover.y})
            </span>
          </div>
          <div className={styles.tooltipRow}>
            <span style={{ color: 'var(--text-muted)' }}>Pass / fail</span>
            <span
              className={`${styles.badge} ${hover.pass_fail ? styles.badgePass : styles.badgeFail}`}
            >
              {hover.pass_fail ? 'PASS' : 'FAIL'}
            </span>
          </div>
          <div className={styles.tooltipRow}>
            <span style={{ color: 'var(--text-muted)' }}>Thickness</span>
            <span>{hover.film_thickness.toFixed(2)} nm</span>
          </div>
          <div className={styles.tooltipRow}>
            <span style={{ color: 'var(--text-muted)' }}>Defect</span>
            <span>{hover.defect_density.toFixed(5)}</span>
          </div>
          <div className={styles.tooltipRow}>
            <span style={{ color: 'var(--text-muted)' }}>Yield</span>
            <span>{(hover.die_yield * 100).toFixed(2)}%</span>
          </div>
          <div
            className={styles.tooltipRow}
            style={{ marginTop: 8, borderTop: '1px solid var(--border)', paddingTop: 6 }}
          >
            <span style={{ color: 'var(--text-muted)' }}>Map metric</span>
            <span style={{ color: 'var(--accent-teal)' }}>
              {METRIC_LABEL[metric]}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
