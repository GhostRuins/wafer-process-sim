import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Component, type CSSProperties, type ReactNode } from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Brush,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { CorrelationMatrix } from '../components/CorrelationMatrix'
import type { DieData, WaferMetric } from '../components/WaferMap'
import { WaferMap } from '../components/WaferMap'
import { KPIBar } from '../components/KPIBar'
import { PredictionPanel } from '../components/PredictionPanel'
import type { ProcessRun } from '../hooks/useWaferData'
import {
  chartGridProps,
  chartTickProps,
  chartTooltipContentStyle,
} from '../styles/chartTheme'
import {
  computeKpis,
  defaultProcessRunDateRange,
  runsToCsv,
  useWaferData,
  worstDieClusterSummary,
} from '../hooks/useWaferData'
import { DashboardTour, type TourStep } from './DashboardTour'

import styles from './Dashboard.module.css'

function toolStroke(toolId: string): string {
  const u = toolId.toUpperCase()
  if (u.endsWith('A')) return '#00d4aa'
  if (u.endsWith('B')) return '#3b82f6'
  if (u.endsWith('C')) return '#f59e0b'
  return '#64748b'
}

const METRIC_PILLS: { id: WaferMetric; label: string }[] = [
  { id: 'film_thickness', label: 'Thickness' },
  { id: 'defect_density', label: 'Defect' },
  { id: 'die_yield', label: 'Yield' },
]

const TOUR_KEY = 'wafer_dashboard_toured'

const TOUR_STEPS: TourStep[] = [
  {
    targetId: 'kpi-bar',
    title: 'Live Process KPIs',
    body: 'Key fab metrics updated per run. AVG YIELD is the percentage of dies per wafer that pass specification - higher is better. 174 active anomalies means the model flagged those runs as statistically abnormal.',
  },
  {
    targetId: 'panel-wafer',
    title: 'Wafer Map',
    body: 'Spatial die-level view of a single wafer run. Each square is one die - color encodes thickness, defect density, or yield. The dashed ring is the edge exclusion zone where deposition rate drops. Click any die to see its exact measurements.',
  },
  {
    targetId: 'panel-yield',
    title: 'Yield Trend',
    body: 'Wafer yield over 2000 runs, colored by tool. Diamond markers are anomaly-flagged runs. The dashed line is a 10-run rolling average. Drag the brush at the bottom to zoom into any time window.',
  },
  {
    targetId: 'panel-corr',
    title: 'Parameter Correlations',
    body: 'Pearson r between process parameters and yield. GAS->YLD at -0.84 means high gas flow strongly predicts lower yield - this is the turbulent contamination effect from the physics model. Click any cell to see the scatter plot.',
  },
  {
    targetId: 'panel-pred',
    title: 'ML Yield Prediction',
    body: 'Set process parameters and click Predict. The ensemble model (LightGBM + XGBoost + Gaussian Process) returns a yield estimate with a 95% confidence interval and a SHAP breakdown showing which parameters are helping or hurting yield.',
  },
]

type YieldPoint = {
  idx: number
  run_id: string
  wafer_yield: number
  tool_id: string
  anomaly_injected: boolean
  roll10: number | null
}

function YieldShape(props: unknown) {
  const p = props as Readonly<{
    cx?: number
    cy?: number
    payload?: YieldPoint
  }>
  const { cx, cy, payload } = p
  if (cx === undefined || cy === undefined || !payload) return <g />
  if (payload.anomaly_injected) {
    const s = 7
    return (
      <polygon
        points={`${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`}
        fill="#ef4444"
        stroke="#fecaca"
        strokeWidth={0.8}
      />
    )
  }
  return (
    <circle
      cx={cx}
      cy={cy}
      r={6}
      fill={toolStroke(payload.tool_id)}
      stroke="#0a0e1a"
      strokeWidth={1}
    />
  )
}

function DieDetailCard({
  die,
  metric,
}: {
  die: DieData | null
  metric: WaferMetric
}) {
  return (
    <aside className={styles.dieCard}>
      <h3 className={styles.dieCardTitle}>Selected die</h3>
      {!die ? (
        <p
          style={{
            fontSize: 12,
            color: 'var(--text-muted)',
            margin: 0,
            fontFamily: 'var(--font-sans)',
          }}
        >
          Click a die on the wafer map.
        </p>
      ) : (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--text-primary)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-muted)' }}>Coords</span>
            <span>
              ({die.x}, {die.y})
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-muted)' }}>Pass / fail</span>
            <span
              style={{
                color: die.pass_fail ? 'var(--accent-teal)' : 'var(--accent-red)',
                fontWeight: 600,
              }}
            >
              {die.pass_fail ? 'PASS' : 'FAIL'}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-muted)' }}>Thickness</span>
            <span>{die.film_thickness.toFixed(2)} nm</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-muted)' }}>Defect</span>
            <span>{die.defect_density.toFixed(4)}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-muted)' }}>Die yield</span>
            <span>{(die.die_yield * 100).toFixed(2)}%</span>
          </div>
          <div
            style={{
              borderTop: '1px solid var(--border)',
              paddingTop: 8,
              fontSize: 10,
              color: 'var(--text-muted)',
            }}
          >
            Map metric:{' '}
            <span style={{ color: 'var(--accent-teal)' }}>{metric}</span>
          </div>
        </div>
      )}
    </aside>
  )
}

function PanelSkeleton({ className = '' }: { className?: string }) {
  return (
    <div
      className={`skeleton rounded-[8px] border border-[var(--border)] bg-[var(--bg-surface)] p-4 ${className}`}
      style={{ minHeight: 120 }}
    >
      <div
        className="mb-3 h-3 w-28 rounded"
        style={{ background: 'var(--bg-elevated)' }}
      />
      <div
        className="h-40 rounded md:h-48"
        style={{ background: 'var(--bg-elevated)' }}
      />
    </div>
  )
}

class PanelErrorBoundary extends Component<
  { title: string; children: ReactNode },
  { hasError: boolean }
> {
  constructor(props: { title: string; children: ReactNode }) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true }
  }

  override render() {
    if (this.state.hasError) {
      return (
        <div className={styles.alert} role="alert">
          {this.props.title} failed to render. Refresh to retry.
        </div>
      )
    }
    return this.props.children
  }
}

function formatRunDisplay(run: ProcessRun) {
  const lotRaw = Number((run.lot_id ?? 'LOT_000').split('_')[1] ?? 0) + 1
  const lot = String(lotRaw).padStart(3, '0')
  const wafer = String(run.lot_position ?? 0).padStart(2, '0')
  const tool = String(run.tool_id ?? '')
    .replace(/^tool_/i, '')
    .toUpperCase()
  const dt = new Date(run.timestamp)
  const dtLabel = Number.isNaN(dt.getTime())
    ? run.timestamp
    : `${dt.toLocaleString('en-US', {
        month: 'short',
      })} ${String(dt.getDate()).padStart(2, '0')} ${dt.getFullYear()}  ${String(
        dt.getHours(),
      ).padStart(2, '0')}:${String(dt.getMinutes()).padStart(2, '0')}`
  const status = run.anomaly_type || 'nominal'
  return {
    primary: `Lot ${lot} · Wafer ${wafer} · Tool ${tool}`,
    secondary: `${dtLabel} · Yield ${(run.wafer_yield * 100).toFixed(1)}%`,
    status,
    lot,
    tool,
    date: dtLabel.toLowerCase(),
  }
}

function RunSelector({
  runs,
  runIndex,
  setRunIndex,
}: {
  runs: ProcessRun[]
  runIndex: number
  setRunIndex: (i: number) => void
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const selected = runs[runIndex]
  const selectedLabel = selected ? formatRunDisplay(selected).primary : 'Select run'

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const all = runs.map((r, i) => ({ i, r, f: formatRunDisplay(r) }))
    if (!q) return all
    return all.filter(
      ({ f }) =>
        f.primary.toLowerCase().includes(q) ||
        f.secondary.toLowerCase().includes(q) ||
        f.status.toLowerCase().includes(q),
    )
  }, [runs, query])

  const grouped = useMemo(() => {
    const out: Array<{ header: string; items: typeof filtered }> = []
    let current = ''
    for (const item of filtered) {
      const lot = item.f.lot
      const day = item.f.secondary.split('·')[0]?.trim() ?? ''
      const header = `LOT ${lot} · Tool ${item.f.tool} · ${day}`.toUpperCase()
      if (header !== current) {
        out.push({ header, items: [] as typeof filtered })
        current = header
      }
      out[out.length - 1]!.items.push(item)
    }
    return out
  }, [filtered])

  return (
    <div style={{ position: 'relative' }}>
      <button
        type="button"
        className={`${styles.runSelect} interactive`}
        onClick={() => setOpen((v) => !v)}
      >
        {selectedLabel}
      </button>
      {open && (
        <div className={styles.runMenu}>
          <input
            className={styles.runSearch}
            placeholder="Filter by lot, tool, date..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className={styles.runList}>
            {grouped.length === 0 && (
              <div className={styles.runEmpty}>No runs found.</div>
            )}
            {grouped.map((g) => (
              <div key={g.header}>
                <div className={styles.runHeader}>-- {g.header} --</div>
                {g.items.map(({ i, f }) => (
                  <button
                    key={i}
                    type="button"
                    className={`${styles.runItem} ${i === runIndex ? styles.runItemActive : ''}`}
                    onClick={() => {
                      setRunIndex(i)
                      setOpen(false)
                    }}
                  >
                    <div className={styles.runPrimary}>{f.primary}</div>
                    <div className={styles.runSecondary}>
                      {f.secondary}
                      {f.status !== 'nominal' ? (
                        <span className={styles.statusPill}>{f.status}</span>
                      ) : (
                        <span className={styles.statusMuted}>nominal</span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function DashboardInner() {
  const [{ start: dateStart, end: dateEnd }, setRange] = useState(
    () => defaultProcessRunDateRange(),
  )
  const [metric, setMetric] = useState<WaferMetric>('film_thickness')
  const [selectedDie, setSelectedDie] = useState<DieData | null>(null)
  const [tourOpen, setTourOpen] = useState(false)
  const [tourStep, setTourStep] = useState(0)

  const {
    runsQuery,
    diesQuery,
    sortedRuns,
    selectedRun,
    runIndex,
    setRunIndex,
    goPrevRun,
    goNextRun,
  } = useWaferData(dateStart, dateEnd)

  const todayIso = useMemo(() => new Date().toISOString(), [])
  const kpis = useMemo(
    () => computeKpis(sortedRuns, todayIso),
    [sortedRuns, todayIso],
  )

  const activeAnomalies = useMemo(
    () => sortedRuns.filter((r) => r.anomaly_injected).length,
    [sortedRuns],
  )

  const worstCluster = useMemo(() => {
    const dies = diesQuery.data ?? []
    return worstDieClusterSummary(dies)
  }, [diesQuery.data])

  const chartData: YieldPoint[] = useMemo(() => {
    const y = sortedRuns.map((r) => r.wafer_yield)
    const roll = y.map((_, i) => {
      const a = Math.max(0, i - 9)
      const sl = y.slice(a, i + 1)
      return sl.reduce((s, v) => s + v, 0) / sl.length
    })
    return sortedRuns.map((r, idx) => ({
      idx,
      run_id: r.run_id,
      wafer_yield: r.wafer_yield,
      tool_id: r.tool_id,
      anomaly_injected: Boolean(r.anomaly_injected),
      roll10: roll[idx] ?? null,
    }))
  }, [sortedRuns])

  const exportCsv = useCallback(() => {
    const csv = runsToCsv(sortedRuns)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `wafer-runs-${dateStart.slice(0, 10)}_${dateEnd.slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [sortedRuns, dateStart, dateEnd])

  const runsLoading = runsQuery.isLoading
  const runsError = runsQuery.isError
  const diesError = diesQuery.isError
  const dies = diesQuery.data ?? []

  const delay = (ms: number): CSSProperties =>
    ({ '--enter-delay': `${ms}ms` }) as CSSProperties

  useEffect(() => {
    const toured = localStorage.getItem(TOUR_KEY) === '1'
    if (!toured) setTourOpen(true)
  }, [])

  const closeTour = useCallback(() => {
    localStorage.setItem(TOUR_KEY, '1')
    setTourOpen(false)
    setTourStep(0)
  }, [])

  const nextTour = useCallback(() => {
    setTourStep((s) => {
      if (s >= TOUR_STEPS.length - 1) {
        closeTour()
        return s
      }
      return s + 1
    })
  }, [closeTour])

  return (
    <div className={styles.shell}>
      <div className="scanlines" aria-hidden />

      <KPIBar
        onOpenTour={() => {
          setTourOpen(true)
          setTourStep(0)
        }}
        avgYieldTodayPct={
          kpis.avgYieldToday === null ? null : kpis.avgYieldToday * 100
        }
        totalRuns={kpis.totalRuns}
        bestTool={kpis.bestTool}
        worstDieCluster={worstCluster?.label ?? '—'}
        activeAnomalies={activeAnomalies}
        dateStart={dateStart}
        dateEnd={dateEnd}
        onDateStartChange={(v) => setRange((r) => ({ ...r, start: v }))}
        onDateEndChange={(v) => setRange((r) => ({ ...r, end: v }))}
        onExportCsv={exportCsv}
        exportDisabled={sortedRuns.length === 0}
        loading={runsLoading}
      />

      {runsError && (
        <div className={styles.alert} role="alert">
          Failed to load runs: {runsQuery.error?.message}. Check{' '}
          <code>/api/process-runs</code> and the API base URL.
        </div>
      )}
      {diesError && (
        <div className={styles.alert} role="alert">
          Failed to load die map: {diesQuery.error?.message}.
        </div>
      )}

      <div className={styles.main}>
        <section
          id="panel-wafer"
          className={`${styles.wafer} ${styles.panel} panel-enter`}
          style={delay(0)}
        >
          <PanelErrorBoundary title="Wafer explorer panel">
            <div className={styles.panelInner}>
              {runsLoading ? (
                <PanelSkeleton className="min-h-[280px]" />
              ) : (
                <>
                  <div className={styles.explorerHeader}>
                  <span className={styles.explorerTitle}>Wafer explorer</span>
                  <div className={styles.runNav}>
                    <button
                      type="button"
                      className={`${styles.navBtn} interactive`}
                      onClick={goPrevRun}
                      disabled={runIndex <= 0}
                      aria-label="Previous run"
                    >
                      ←
                    </button>
                    <RunSelector
                      runs={sortedRuns}
                      runIndex={runIndex}
                      setRunIndex={(i) => setRunIndex(i)}
                    />
                    <button
                      type="button"
                      className={`${styles.navBtn} interactive`}
                      onClick={goNextRun}
                      disabled={runIndex >= sortedRuns.length - 1}
                      aria-label="Next run"
                    >
                      →
                    </button>
                  </div>
                  <div className={styles.metricPills}>
                    {METRIC_PILLS.map((p) => (
                      <button
                        key={p.id}
                        type="button"
                        className={`${styles.pill} interactive ${metric === p.id ? styles.pillActive : ''}`}
                        onClick={() => setMetric(p.id)}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>

                  <div className={styles.waferBody}>
                    <div className={styles.mapCol}>
                      {selectedRun && dies.length > 0 ? (
                        <WaferMap
                          wafer_id={selectedRun.wafer_id}
                          metric={metric}
                          data={dies}
                          onDieClick={setSelectedDie}
                          showExclusionZone
                        />
                      ) : diesQuery.isLoading && selectedRun ? (
                        <PanelSkeleton className="min-h-[260px] flex-1" />
                      ) : (
                        <div
                          className="flex flex-1 items-center justify-center rounded-[8px] border border-dashed p-6 text-center"
                          style={{
                            borderColor: 'var(--border)',
                            color: 'var(--text-muted)',
                            fontSize: 12,
                          }}
                        >
                          No die map for this run. Endpoint{' '}
                          <code className="font-mono text-[var(--accent-teal)]">
                            /api/runs/&lt;run_id&gt;/dies
                          </code>
                        </div>
                      )}
                    </div>
                    <div className={styles.dieCol}>
                      <DieDetailCard die={selectedDie} metric={metric} />
                    </div>
                  </div>
                </>
              )}
            </div>
          </PanelErrorBoundary>
        </section>

        <section
          id="panel-yield"
          className={`${styles.yield} ${styles.panel} panel-enter`}
          style={delay(100)}
        >
          <PanelErrorBoundary title="Yield trend panel">
            <div className={styles.panelInner}>
              <h3 className={styles.panelTitle}>Yield trend</h3>
              <p className={styles.panelSubtitle}>
                Run order · tools A/B/C · ◆ anomaly · dashed = 10-run avg
              </p>
              {runsLoading ? (
                <PanelSkeleton className="mt-1 border-0 bg-transparent p-0" />
              ) : chartData.length === 0 ? (
                <p
                  className="mt-8 text-center"
                  style={{ color: 'var(--text-muted)', fontSize: 12 }}
                >
                  No runs in range.
                </p>
              ) : (
                <div className={styles.chartFill}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                      data={chartData}
                      margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
                    >
                    <CartesianGrid {...chartGridProps} />
                    <XAxis
                      dataKey="idx"
                      type="number"
                      stroke="var(--border)"
                      tick={chartTickProps}
                      label={{
                        value: 'Run index',
                        position: 'insideBottom',
                        offset: -4,
                        fill: 'var(--text-muted)',
                        fontSize: 11,
                      }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      stroke="var(--border)"
                      tick={chartTickProps}
                      tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
                    />
                    <Tooltip
                      contentStyle={chartTooltipContentStyle}
                      formatter={(value, name) => {
                        const n = String(name ?? '')
                        const v = Number(value)
                        if (n === 'roll10')
                          return [`${(v * 100).toFixed(2)}%`, '10-run avg']
                        if (n === 'wafer_yield')
                          return [`${(v * 100).toFixed(2)}%`, 'Yield']
                        return [String(value ?? ''), n]
                      }}
                      labelFormatter={(_, payload) =>
                        payload?.[0]?.payload?.run_id ?? ''
                      }
                    />
                    <Brush
                      dataKey="idx"
                      height={20}
                      stroke="#00d4aa"
                      fill="rgba(26, 34, 53, 0.5)"
                      travellerWidth={8}
                    />
                    <Line
                      type="monotone"
                      dataKey="roll10"
                      stroke="#f1f5f9"
                      strokeWidth={1}
                      strokeDasharray="4 4"
                      dot={false}
                      connectNulls
                    />
                    <Scatter
                      dataKey="wafer_yield"
                      fill="#8884d8"
                      shape={YieldShape}
                    />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </PanelErrorBoundary>
        </section>

        <section
          id="panel-corr"
          className={`${styles.corr} ${styles.panel} panel-enter`}
          style={delay(200)}
        >
          <PanelErrorBoundary title="Correlation matrix panel">
            <div className={styles.panelInner}>
              <CorrelationMatrix runs={sortedRuns} loading={runsLoading} />
            </div>
          </PanelErrorBoundary>
        </section>

        <section
          id="panel-pred"
          className={`${styles.pred} ${styles.panel} panel-enter`}
          style={delay(300)}
        >
          <PanelErrorBoundary title="Prediction panel">
            <div className={styles.panelInner}>
              <PredictionPanel seedFromRun={selectedRun} loading={runsLoading} />
            </div>
          </PanelErrorBoundary>
        </section>
      </div>
      <DashboardTour
        open={tourOpen}
        steps={TOUR_STEPS}
        stepIndex={tourStep}
        onNext={nextTour}
        onSkip={closeTour}
      />
    </div>
  )
}

export function Dashboard() {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60_000,
            retry: 1,
          },
        },
      }),
  )

  return (
    <QueryClientProvider client={client}>
      <DashboardInner />
    </QueryClientProvider>
  )
}
