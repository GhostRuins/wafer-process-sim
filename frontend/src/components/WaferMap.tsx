import {
  type ReactElement,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'

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
  onDieClick?: (die: DieData | null) => void
}

const FALLBACK_VB = 400

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
  const wrapRef = useRef<HTMLDivElement>(null)
  const [vb, setVb] = useState(FALLBACK_VB)

  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el || typeof ResizeObserver === 'undefined') return

    const update = () => {
      const r = el.getBoundingClientRect()
      const d = Math.floor(Math.max(64, Math.min(r.width, r.height)))
      setVb((prev) => (prev === d ? prev : d))
    }

    update()
    const ro = new ResizeObserver(() => update())
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const cx = vb / 2
  const cy = vb / 2
  const R = (vb / 2) * 0.9

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
  }, [data, metric, R])

  const { min, max, mean, stdev } = useMemo(() => {
    if (vals.length === 0) {
      return { min: 0, max: 1, mean: 0, stdev: 0 }
    }
    const min = Math.min(...vals)
    const max = Math.max(...vals)
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length
    const v =
      vals.length > 1
        ? vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length
        : 0
    const stdev = Math.sqrt(v)
    return { min, max, mean, stdev }
  }, [vals])

  const colorFor = useMemo(
    () => createMetricColorScale(colorScale, [min, max]),
    [colorScale, min, max],
  )

  const cells = useMemo(() => {
    const out: ReactElement[] = []
    for (const item of normalizedDies) {
      const die = item.die
      const v = metricValue(die, metric)
      const fill = colorFor(v)
      const dieCx = cx + item.xNorm * R
      const dieCy = cy + item.yNorm * R
      const left = dieCx - dieSizePx / 2
      const top = dieCy - dieSizePx / 2
      out.push(
        <g
          key={item.key}
          transform={`translate(${dieCx},${dieCy})`}
        >
          <rect
            x={left - dieCx}
            y={top - dieCy}
            width={dieSizePx}
            height={dieSizePx}
            fill={fill}
            stroke="var(--bg-primary)"
            strokeWidth={0.5}
            className="cursor-pointer interactive"
            onPointerEnter={() => onDieClick?.(die)}
            onPointerLeave={() => onDieClick?.(null)}
          />
          {!die.pass_fail && (
            <rect
              x={left - dieCx}
              y={top - dieCy}
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
    cx,
    cy,
    R,
  ])

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
      <div className={styles.mapRow}>
        <div ref={wrapRef} className={styles.svgWrap}>
          <svg
            viewBox={`0 0 ${vb} ${vb}`}
            preserveAspectRatio="xMidYMid meet"
            className={styles.svg}
            role="img"
            aria-label={`Wafer map for ${wafer_id}`}
          >
            <defs>
              <clipPath id={clipId}>
                <circle cx={cx} cy={cy} r={R} />
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
              cx={cx}
              cy={cy}
              r={R}
              fill="var(--bg-primary)"
              stroke="var(--border)"
              strokeWidth={1}
            />
            <g clipPath={`url(#${clipId})`}>{cells}</g>
            {showExclusionZone && (
              <circle
                cx={cx}
                cy={cy}
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
          <p className={styles.legendMetric}>{METRIC_LABEL[metric]}</p>
        </div>
      </div>

      <div className={styles.statsRow} aria-label="Map metric statistics">
        <span>min {min.toFixed(3)}</span>
        <span className={styles.statsSep}>·</span>
        <span>max {max.toFixed(3)}</span>
        <span className={styles.statsSep}>·</span>
        <span>μ {mean.toFixed(3)}</span>
        <span className={styles.statsSep}>·</span>
        <span>σ {stdev.toFixed(3)}</span>
      </div>
    </div>
  )
}
