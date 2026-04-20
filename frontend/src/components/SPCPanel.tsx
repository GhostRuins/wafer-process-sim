import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { CapabilityPanel } from './CapabilityPanel'
import { useSpcData, type SpcMetric, type SpcMode, type SpcReplaySpeed } from '../hooks/useSpcData'
import type { ProcessRun } from '../hooks/useWaferData'
import { chartGridProps, chartTickProps } from '../styles/chartTheme'
import styles from '../pages/Dashboard.module.css'

const METRIC_LABELS: Record<SpcMetric, string> = {
  wafer_yield: 'Wafer yield',
  mean_film_thickness_nm: 'Mean thickness',
  mean_defect_density_cm2: 'Mean defect density',
}

const NELSON_RULE_DESCRIPTIONS: Record<string, string> = {
  R1: 'Point beyond 3sigma',
  R2: '9 points same side of mean',
  R3: '6 points trending',
  R4: '14 alternating up/down',
  R5: '2 of 3 points beyond 2sigma',
  R6: '4 of 5 points beyond 1sigma',
  R7: '15 points within 1sigma',
  R8: '8 points beyond 1sigma both sides',
}

const formatMetricValue = (value: number | null | undefined, metric: SpcMetric): string => {
  if (value === undefined || value === null || Number.isNaN(value)) return '—'
  switch (metric) {
    case 'wafer_yield':
      return `${(value * 100).toFixed(2)}%`
    case 'mean_film_thickness_nm':
      return `${value.toFixed(2)} nm`
    case 'mean_defect_density_cm2':
      return `${value.toFixed(4)} cm^-2`
    default:
      return value.toFixed(4)
  }
}

export function SPCPanel({
  start,
  end,
  tools,
  runs,
  runsLoading,
}: {
  start: string
  end: string
  tools: string[]
  runs: ProcessRun[]
  runsLoading: boolean
}) {
  const [metric, setMetric] = useState<SpcMetric>('wafer_yield')
  const [mode, setMode] = useState<SpcMode>('batch')
  const [replaySpeed, setReplaySpeed] = useState<SpcReplaySpeed>('1x')
  const [toolId, setToolId] = useState('')
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const { batchQuery, points, violations, summary, wsStatus, totalRuns, streamFinished, resetStream, pauseStream, resumeStream, stopStream, setStreamSpeed } = useSpcData({
    start,
    end,
    metric,
    toolId: toolId || undefined,
    mode,
    replaySpeed,
  })

  const selectedViolation = useMemo(() => (selectedIdx === null ? null : violations[selectedIdx] ?? null), [selectedIdx, violations])
  const violationIndices = useMemo(() => new Set(violations.map((v) => v.index)), [violations])
  const violationPoints = useMemo(() => points.filter((p) => violationIndices.has(p.index)), [points, violationIndices])
  const xDomain = useMemo(() => [0, Math.max(points.length + 10, 50)], [points.length])
  const progressRatio = totalRuns > 0 ? Math.min(1, points.length / totalRuns) : 0
  const violationsByIndex = useMemo(() => {
    const map = new Map<number, typeof violations>()
    for (const v of violations) {
      map.set(v.index, [...(map.get(v.index) ?? []), v])
    }
    return map
  }, [violations])

  const SPCTooltip = (props: { active?: boolean; payload?: Array<{ payload?: unknown }> }) => {
    const { active, payload } = props
    if (!active || !payload?.length) return null
    const d = payload[0]?.payload as
      | {
          index?: number
          run_index?: number
          lot_label?: string
          value?: number
          ewma?: number
          tool_id?: string | null
          timestamp?: string
          anomaly_type?: string | null
          is_anomaly?: boolean
          violated_rules?: string[]
          is_violation?: boolean
        }
      | undefined
    if (!d) return null

    const idx = d.index ?? d.run_index ?? 0
    const relatedViolations = violationsByIndex.get(idx) ?? []
    const violatedRules = relatedViolations.map((v) => `R${v.rule_id}`)
    const isViolation = relatedViolations.length > 0 || Boolean(d.is_violation)
    const anomalyType = d.anomaly_type ?? null
    const isAnomaly = Boolean(d.is_anomaly || anomalyType)

    return (
      <div
        style={{
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border)',
          borderRadius: '6px',
          padding: '10px 14px',
          fontSize: '11px',
          fontFamily: 'var(--font-mono)',
          minWidth: '200px',
        }}
      >
        <div
          style={{
            color: 'var(--text-muted)',
            marginBottom: '6px',
            fontFamily: 'var(--font-sans)',
            fontSize: '10px',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}
        >
          {d.lot_label ?? `Run ${idx}`}
        </div>
        <div style={{ color: 'var(--accent-teal)', fontSize: '16px', marginBottom: '8px' }}>
          {formatMetricValue(d.value, metric)}
        </div>
        <div style={{ color: '#f59e0b', marginBottom: '4px' }}>
          EWMA: {formatMetricValue(d.ewma, metric)}
        </div>
        {d.tool_id ? (
          <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>
            Tool: {d.tool_id.replace('tool_', 'Tool ')}
          </div>
        ) : (
          <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>Tool: —</div>
        )}
        {d.timestamp ? (
          <div style={{ color: 'var(--text-muted)', marginBottom: '8px' }}>
            {new Date(d.timestamp).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
            })}
          </div>
        ) : null}
        {isViolation && violatedRules?.length > 0 ? (
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: '8px', marginTop: '4px' }}>
            <div
              style={{
                color: '#ef4444',
                fontFamily: 'var(--font-sans)',
                fontSize: '10px',
                textTransform: 'uppercase',
                marginBottom: '4px',
              }}
            >
              SPC Violations
            </div>
            {violatedRules.map((rule) => (
              <div key={`${idx}-${rule}`} style={{ color: '#ef4444', marginBottom: '2px' }}>
                {rule}: {NELSON_RULE_DESCRIPTIONS[rule] ?? 'Rule triggered'}
              </div>
            ))}
          </div>
        ) : null}
        {isAnomaly && anomalyType ? (
          <div
            style={{
              borderTop: '1px solid var(--border)',
              paddingTop: '8px',
              marginTop: '4px',
              color: '#f59e0b',
              fontFamily: 'var(--font-sans)',
              fontSize: '10px',
            }}
          >
            Ground truth: {anomalyType.replace('_', ' ')}
          </div>
        ) : null}
      </div>
    )
  }

  return (
    <div className={styles.spcRoot}>
      <h3 className={styles.panelTitle}>SPC mode</h3>
      <div className={styles.spcControls}>
        <select value={mode} onChange={(e) => setMode(e.target.value as SpcMode)} className={styles.runSelect}>
          <option value="batch">Batch</option>
          <option value="stream">Stream</option>
        </select>
        {mode === 'stream' && (
          <span className={styles.statusMuted}>WS: {wsStatus}</span>
        )}
        <select value={metric} onChange={(e) => setMetric(e.target.value as SpcMetric)} className={styles.runSelect}>
          {Object.entries(METRIC_LABELS).map(([k, v]) => (
            <option key={k} value={k}>
              {v}
            </option>
          ))}
        </select>
        <select value={toolId} onChange={(e) => setToolId(e.target.value)} className={styles.runSelect}>
          <option value="">All tools</option>
          {tools.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </div>
      {mode === 'stream' && (
        <div className={styles.spcPlaybackBar}>
          <button
            type="button"
            className={styles.spcPlayBtn}
            onClick={() => {
              if (wsStatus === 'live') pauseStream()
              else resumeStream()
            }}
          >
            {wsStatus === 'live' ? '⏸ Pause' : '▶ Play'}
          </button>
          <button type="button" className={styles.spcStopBtn} onClick={stopStream}>
            ⏹ Stop
          </button>
          <button
            type="button"
            className={`${styles.spcReplayBtn} ${streamFinished ? styles.spcReplayBtnDone : ''}`}
            onClick={() => {
              resetStream()
              setSelectedIdx(null)
            }}
          >
            ↺ Replay
          </button>
          <div className={styles.spcProgressWrap}>
            <div className={styles.spcProgressText}>
              {streamFinished ? `Stream complete - ${points.length}/${Math.max(totalRuns, points.length)} runs` : `run ${points.length}/${Math.max(totalRuns, points.length || 0)}`}
            </div>
            <div className={styles.spcProgressTrack}>
              <div className={styles.spcProgressFill} style={{ width: `${progressRatio * 100}%` }} />
            </div>
          </div>
          <div className={styles.spcSpeedPills}>
            {(['0.5x', '1x', '5x', '10x', 'max'] as const).map((s) => (
              <button
                key={s}
                type="button"
                className={`${styles.spcSpeedPill} ${replaySpeed === s ? styles.spcSpeedPillActive : ''}`}
                onClick={() => {
                  setReplaySpeed(s as SpcReplaySpeed)
                  setStreamSpeed(s as SpcReplaySpeed)
                }}
              >
                {s === 'max' ? 'Max' : s}
              </button>
            ))}
          </div>
        </div>
      )}
      {batchQuery.isError && <p className={styles.alert}>SPC load failed: {batchQuery.error?.message}</p>}

      <div className={styles.spcChartFill}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={points} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid {...chartGridProps} />
            <XAxis dataKey="index" stroke="var(--border)" tick={chartTickProps} domain={xDomain} type="number" />
            <YAxis
              stroke="var(--border)"
              tick={chartTickProps}
              domain={['dataMin - 0.02', 'dataMax + 0.02']}
              allowDataOverflow={false}
            />
            <Tooltip content={<SPCTooltip />} />
            <ReferenceLine y={points[0]?.cl} stroke="#f1f5f9" strokeDasharray="4 4" />
            <ReferenceLine y={points[0]?.ucl} stroke="#f97316" strokeDasharray="3 3" />
            <ReferenceLine y={points[0]?.lcl} stroke="#f97316" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="value" stroke="#00d4aa" dot={false} strokeWidth={1.3} isAnimationActive={false} />
            <Line type="monotone" dataKey="ewma" stroke="#60a5fa" dot={false} strokeWidth={1} isAnimationActive={false} />
            <Scatter data={violationPoints} dataKey="value" fill="#ef4444" shape="diamond" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className={styles.spcSummaryRow}>
        <span>Total points: {summary?.total_points ?? points.length}</span>
        <span>Violations: {summary?.total_violations ?? violations.length}</span>
        {summary?.f1 !== undefined && summary.f1 !== null && <span>F1: {summary.f1.toFixed(3)}</span>}
      </div>

      <CapabilityPanel
        runs={runs}
        loading={runsLoading}
        dateStart={start}
        dateEnd={end}
        spcToolFilter={toolId}
      />

      <div className={styles.spcAlerts}>
        <h4 className={styles.panelSubtitle}>Alert feed</h4>
        {violations.slice(0, 20).map((v, idx) => (
          <button
            key={`${v.rule_id}-${v.timestamp}-${idx}`}
            type="button"
            className={`${styles.runItem} ${selectedIdx === idx ? styles.runItemActive : ''}`}
            onClick={() => setSelectedIdx(idx)}
          >
            <div className={styles.runPrimary}>
              R{v.rule_id}: {v.rule_name}
            </div>
            <div className={styles.runSecondary}>
              {new Date(v.timestamp).toLocaleString()} · {v.severity} · {v.value.toFixed(4)}
            </div>
          </button>
        ))}
      </div>
      <div className={styles.spcExplain}>
        <h4 className={styles.panelSubtitle}>Why this alert fired</h4>
        {!selectedViolation ? (
          <p className={styles.statusMuted}>Select an alert from the feed.</p>
        ) : (
          <p className={styles.statusMuted}>
            {selectedViolation.message} ({selectedViolation.rule_name}) at index {selectedViolation.index}. Evidence points:{' '}
            {selectedViolation.evidence_indices.join(', ')}
          </p>
        )}
      </div>
    </div>
  )
}
