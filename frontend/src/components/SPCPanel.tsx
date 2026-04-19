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

import { apiUrl } from '../config'
import { useSpcData, type SpcMetric, type SpcMode } from '../hooks/useSpcData'
import { chartGridProps, chartTickProps, chartTooltipContentStyle } from '../styles/chartTheme'
import styles from '../pages/Dashboard.module.css'

const METRIC_LABELS: Record<SpcMetric, string> = {
  wafer_yield: 'Wafer yield',
  mean_film_thickness_nm: 'Mean thickness',
  mean_defect_density_cm2: 'Mean defect density',
}

export function SPCPanel({ start, end, tools }: { start: string; end: string; tools: string[] }) {
  const [metric, setMetric] = useState<SpcMetric>('wafer_yield')
  const [mode, setMode] = useState<SpcMode>('batch')
  const [toolId, setToolId] = useState('')
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const [replayBusy, setReplayBusy] = useState(false)
  const [replayError, setReplayError] = useState<string | null>(null)
  const { batchQuery, points, violations, summary, wsStatus } = useSpcData({
    start,
    end,
    metric,
    toolId: toolId || undefined,
    mode,
  })

  const selectedViolation = useMemo(() => (selectedIdx === null ? null : violations[selectedIdx] ?? null), [selectedIdx, violations])
  const violationIndices = useMemo(() => new Set(violations.map((v) => v.index)), [violations])
  const violationPoints = useMemo(() => points.filter((p) => violationIndices.has(p.index)), [points, violationIndices])

  const startReplay = async () => {
    setReplayBusy(true)
    setReplayError(null)
    try {
      const res = await fetch(apiUrl('/api/spc/stream/replay'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          metric,
          start,
          end,
          tool_id: toolId || null,
          replay_speed: '10x',
          reset_state: true,
        }),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => '')
        throw new Error(text || `Replay request failed: ${res.status}`)
      }
      setSelectedIdx(null)
    } catch (err) {
      setReplayError(err instanceof Error ? err.message : 'Unknown replay error')
    } finally {
      setReplayBusy(false)
    }
  }

  return (
    <div className={styles.spcRoot}>
      <h3 className={styles.panelTitle}>SPC mode</h3>
      <div className={styles.spcControls}>
        <select value={mode} onChange={(e) => setMode(e.target.value as SpcMode)} className={styles.runSelect}>
          <option value="batch">Batch</option>
          <option value="stream">Stream</option>
        </select>
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
        {mode === 'stream' && <span className={styles.statusMuted}>WS: {wsStatus}</span>}
        {mode === 'stream' && (
          <button type="button" className={styles.navBtn} onClick={startReplay} disabled={replayBusy}>
            {replayBusy ? 'Replaying...' : 'Start replay'}
          </button>
        )}
      </div>
      {batchQuery.isError && <p className={styles.alert}>SPC load failed: {batchQuery.error?.message}</p>}
      {replayError && <p className={styles.alert}>Replay failed: {replayError}</p>}

      <div className={styles.spcChartFill}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={points} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid {...chartGridProps} />
            <XAxis dataKey="index" stroke="var(--border)" tick={chartTickProps} />
            <YAxis
              stroke="var(--border)"
              tick={chartTickProps}
              domain={['dataMin - 0.02', 'dataMax + 0.02']}
              allowDataOverflow={false}
            />
            <Tooltip
              contentStyle={chartTooltipContentStyle}
              formatter={(value, name) => [String(value), String(name)]}
              labelFormatter={(_, p) => (p?.[0]?.payload?.timestamp as string) ?? ''}
            />
            <ReferenceLine y={points[0]?.cl} stroke="#f1f5f9" strokeDasharray="4 4" />
            <ReferenceLine y={points[0]?.ucl} stroke="#f97316" strokeDasharray="3 3" />
            <ReferenceLine y={points[0]?.lcl} stroke="#f97316" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="value" stroke="#00d4aa" dot={false} strokeWidth={1.3} />
            <Line type="monotone" dataKey="ewma" stroke="#60a5fa" dot={false} strokeWidth={1} />
            <Scatter data={violationPoints} dataKey="value" fill="#ef4444" shape="diamond" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className={styles.spcSummaryRow}>
        <span>Total points: {summary?.total_points ?? points.length}</span>
        <span>Violations: {summary?.total_violations ?? violations.length}</span>
        {summary?.f1 !== undefined && summary.f1 !== null && <span>F1: {summary.f1.toFixed(3)}</span>}
      </div>
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
