import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { ProcessRun } from '../hooks/useWaferData'
import {
  type PredictResponse,
  TOOL_ID_FROM_CHOICE,
  type ToolChoice,
  useYieldPrediction,
} from '../hooks/useYieldPrediction'
import {
  chartGridProps,
  chartTickProps,
  chartTooltipContentStyle,
} from '../styles/chartTheme'
import { apiUrl } from '../config'

import styles from './PredictionPanel.module.css'

const PARAM_KEYS = [
  'temperature',
  'pressure',
  'gas_flow',
  'rf_power',
  'deposition_time',
] as const

type ParamKey = (typeof PARAM_KEYS)[number]
type ActiveTab = 'predict' | 'optimize'

type OptimizeResult = {
  optimized_yield: number
  delta_from_start: number
  confidence_interval: [number, number]
  suggested_params: Record<ParamKey, number>
  locked_params: Record<string, number>
  n_evaluations: number
}

const DEFAULT_BOUNDS: Record<ParamKey, [number, number]> = {
  temperature: [350, 450],
  pressure: [20, 100],
  gas_flow: [40, 120],
  rf_power: [150, 300],
  deposition_time: [90, 180],
}

const PARAM_LABEL: Record<ParamKey, string> = {
  temperature: 'Temperature (°C)',
  pressure: 'Pressure (mTorr)',
  gas_flow: 'Gas flow (sccm)',
  rf_power: 'RF power (W)',
  deposition_time: 'Deposition time (s)',
}

function clamp(n: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, n))
}

async function postPredictRaw(body: {
  temperature: number
  pressure: number
  gas_flow: number
  rf_power: number
  deposition_time: number
  tool_id: string
}) {
  const res = await fetch(apiUrl('/api/predict'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Predict failed: ${res.status}`)
  }
  const json = (await res.json()) as {
    predicted_yield: number
    confidence_interval?: [number, number]
  }
  const [lo, hi] = json.confidence_interval ?? [json.predicted_yield, json.predicted_yield]
  return { predicted_yield: json.predicted_yield, ci_low: lo, ci_high: hi }
}

function riskClass(flag: string): string {
  const u = flag.toLowerCase()
  if (
    u.includes('critical') ||
    u.includes('fail') ||
    u.includes('high') ||
    u.includes('severe')
  )
    return styles.riskCrit
  if (u.includes('warn') || u.includes('moderate') || u.includes('caution'))
    return styles.riskWarn
  return styles.riskInfo
}

export function PredictionPanel({
  seedFromRun,
  loading,
}: {
  seedFromRun: ProcessRun | undefined
  loading?: boolean
}) {
  const predict = useYieldPrediction()
  const [tool, setTool] = useState<ToolChoice>('A')
  const [activeTab, setActiveTab] = useState<ActiveTab>('predict')
  const [lockedParams, setLockedParams] = useState<Set<ParamKey>>(
    () => new Set<ParamKey>(),
  )
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [optimizeResult, setOptimizeResult] = useState<OptimizeResult | null>(null)
  const [optimizeProgress, setOptimizeProgress] = useState<{
    current: number
    total: number
  } | null>(null)
  const [inlineWarning, setInlineWarning] = useState<string | null>(null)
  const [toastMessage, setToastMessage] = useState<string | null>(null)
  const [displayedOptimizeYield, setDisplayedOptimizeYield] = useState<number | null>(
    null,
  )
  const [showDelta, setShowDelta] = useState(false)
  const [valueDrafts, setValueDrafts] = useState<Partial<Record<ParamKey, string>>>({})
  const optimizeStartRef = useRef<number | null>(null)
  const [params, setParams] = useState<Record<ParamKey, number>>(() => ({
    temperature: 400,
    pressure: 50,
    gas_flow: 80,
    rf_power: 225,
    deposition_time: 120,
  }))

  useEffect(() => {
    if (!seedFromRun) return
    setParams({
      temperature: seedFromRun.temperature,
      pressure: seedFromRun.pressure,
      gas_flow: seedFromRun.gas_flow,
      rf_power: seedFromRun.rf_power,
      deposition_time: seedFromRun.deposition_time,
    })
    const t = seedFromRun.tool_id?.toUpperCase() ?? 'A'
    if (t.endsWith('A')) setTool('A')
    else if (t.endsWith('B')) setTool('B')
    else if (t.endsWith('C')) setTool('C')
  }, [seedFromRun])

  useEffect(() => {
    if (!toastMessage) return
    const id = window.setTimeout(() => setToastMessage(null), 2600)
    return () => window.clearTimeout(id)
  }, [toastMessage])

  useEffect(() => {
    if (!optimizeResult || optimizeStartRef.current === null) return
    const start = optimizeStartRef.current
    const end = optimizeResult.optimized_yield
    const durationMs = 800
    let rafId = 0
    let startAt = 0
    setShowDelta(false)
    const tick = (ts: number) => {
      if (!startAt) startAt = ts
      const p = clamp((ts - startAt) / durationMs, 0, 1)
      const eased = 1 - Math.pow(1 - p, 3)
      setDisplayedOptimizeYield(start + (end - start) * eased)
      if (p < 1) {
        rafId = window.requestAnimationFrame(tick)
      } else {
        window.setTimeout(() => setShowDelta(true), 200)
      }
    }
    rafId = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(rafId)
  }, [optimizeResult])

  const onPredict = () => {
    predict.mutate({
      temperature: params.temperature,
      pressure: params.pressure,
      gas_flow: params.gas_flow,
      rf_power: params.rf_power,
      deposition_time: params.deposition_time,
      tool_id: TOOL_ID_FROM_CHOICE[tool],
    })
  }

  const toggleLock = (key: ParamKey) => {
    setLockedParams((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const commitParamValue = (key: ParamKey) => {
    const draft = valueDrafts[key]
    if (draft === undefined) return
    const [lo, hi] = DEFAULT_BOUNDS[key]
    const parsed = Number(draft)
    if (Number.isFinite(parsed)) {
      setParams((prev) => ({
        ...prev,
        [key]: clamp(parsed, lo, hi),
      }))
    }
    setValueDrafts((prev) => {
      const next = { ...prev }
      delete next[key]
      return next
    })
  }

  const optimizeClientFallback = async (
    start: Record<ParamKey, number>,
    freeParams: ParamKey[],
    lockedMap: Record<string, number>,
  ): Promise<OptimizeResult> => {
    const iterations = 3
    const maxSamples = Math.max(8, Math.min(12, freeParams.length * iterations * 2))
    let evaluations = 0
    let current = { ...start }
    let bestPred = await postPredictRaw({ ...current, tool_id: TOOL_ID_FROM_CHOICE[tool] })
    evaluations += 1
    setOptimizeProgress({ current: evaluations, total: maxSamples })
    for (let i = 0; i < iterations; i += 1) {
      for (const key of freeParams) {
        const [lo, hi] = DEFAULT_BOUNDS[key]
        const step = (hi - lo) / 12
        const down = clamp(current[key] - step, lo, hi)
        const up = clamp(current[key] + step, lo, hi)
        const candidates: number[] = [down, up]
        for (const candidate of candidates) {
          if (evaluations >= maxSamples) break
          const trial = { ...current, [key]: candidate }
          const pred = await postPredictRaw({
            ...trial,
            tool_id: TOOL_ID_FROM_CHOICE[tool],
          })
          evaluations += 1
          setOptimizeProgress({ current: evaluations, total: maxSamples })
          if (pred.predicted_yield > bestPred.predicted_yield) {
            bestPred = pred
            current = trial
          }
        }
      }
    }
    return {
      optimized_yield: bestPred.predicted_yield,
      delta_from_start: bestPred.predicted_yield - (optimizeStartRef.current ?? 0),
      confidence_interval: [bestPred.ci_low, bestPred.ci_high],
      suggested_params: current,
      locked_params: lockedMap,
      n_evaluations: evaluations,
    }
  }

  const onOptimize = async () => {
    const lockCount = lockedParams.size
    const warning =
      'Lock at least one variable to constrain the optimization. Without constraints, the optimizer will always return the same global optimum regardless of your settings.'
    if (lockCount === 0) {
      setInlineWarning(warning)
      setToastMessage(warning)
      return
    }
    setInlineWarning(null)
    setOptimizeResult(null)
    setOptimizeProgress(null)
    setIsOptimizing(true)
    const freeParams = PARAM_KEYS.filter((k) => !lockedParams.has(k))
    const startValues = { ...params }
    try {
      const startPred = await postPredictRaw({
        ...startValues,
        tool_id: TOOL_ID_FROM_CHOICE[tool],
      })
      optimizeStartRef.current = startPred.predicted_yield
      const lockedMap = Object.fromEntries(
        [...lockedParams].map((k) => [k, startValues[k]]),
      )
      const payload = {
        locked_params: lockedMap,
        free_params: freeParams,
        tool_id: TOOL_ID_FROM_CHOICE[tool],
        starting_point: Object.fromEntries(freeParams.map((k) => [k, startValues[k]])),
        target: 'maximize_yield',
      }
      let result: OptimizeResult
      const optimizeRes = await fetch(apiUrl('/api/optimize'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (optimizeRes.ok) {
        const json = (await optimizeRes.json()) as OptimizeResult
        result = {
          ...json,
          suggested_params: {
            ...startValues,
            ...json.suggested_params,
          },
          delta_from_start: json.optimized_yield - startPred.predicted_yield,
        }
      } else {
        result = await optimizeClientFallback(startValues, freeParams, lockedMap)
      }
      setOptimizeResult(result)
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Optimization failed'
      setToastMessage(message)
    } finally {
      setOptimizeProgress(null)
      setIsOptimizing(false)
    }
  }

  const shapBars = useMemo(() => {
    const data = predict.data as PredictResponse | undefined
    if (!data?.shap_values?.length) return []
    return [...data.shap_values]
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .map((s) => ({ name: s.feature, value: s.value }))
  }, [predict.data])

  if (loading) {
    return (
      <div className={styles.root}>
        <div className={`${styles.skeleton} skeleton`} />
      </div>
    )
  }

  const err = predict.error as Error | null
  const data = predict.data as PredictResponse | undefined
  const busy = predict.isPending
  const lockCount = lockedParams.size
  const freeCount = PARAM_KEYS.length - lockCount
  const zeroLocks = lockCount === 0
  const tabTransitionClass =
    activeTab === 'optimize' ? styles.optimizeActive : styles.predictActive

  return (
    <div className={styles.root}>
      {toastMessage && <div className={styles.toast}>{toastMessage}</div>}
      <div className={styles.head}>
        <h3 className={styles.title}>Yield prediction</h3>
        <span
          className={`${styles.dot} ${busy ? styles.dotBusy : styles.dotReady}`}
          aria-hidden
        />
      </div>
      <p className={styles.subtitle}>
        Predicted % of dies passing spec · pre-electrical-test estimate
      </p>
      <div className={styles.tabs}>
        <button
          type="button"
          className={`${styles.tabBtn} ${activeTab === 'predict' ? styles.tabBtnActive : ''}`}
          onClick={() => setActiveTab('predict')}
        >
          PREDICT
        </button>
        <button
          type="button"
          className={`${styles.tabBtn} ${activeTab === 'optimize' ? styles.tabBtnActive : ''}`}
          onClick={() => setActiveTab('optimize')}
        >
          OPTIMIZE
        </button>
      </div>

      <select
        className={styles.toolSelect}
        value={tool}
        onChange={(e) => setTool(e.target.value as ToolChoice)}
      >
        <option value="A">Tool A (tool_A)</option>
        <option value="B">Tool B (tool_B)</option>
        <option value="C">Tool C (tool_C)</option>
      </select>

      {activeTab === 'optimize' && optimizeResult && (
        <div className={`${styles.optimizeResultCard} ${styles.optimizeResultEnter}`}>
          <p className={styles.optimizeResultLabel}>OPTIMIZED YIELD</p>
          <div className={styles.optimizeYieldRow}>
            <p className={styles.optimizeYieldBig}>
              {((displayedOptimizeYield ?? optimizeResult.optimized_yield) * 100).toFixed(1)}%
            </p>
            <span className={styles.optimizeYieldUnit}>die yield</span>
            <span
              className={`${styles.optimizeDelta} ${showDelta ? styles.optimizeDeltaShow : ''} ${optimizeResult.delta_from_start >= 0 ? styles.deltaUp : styles.deltaDown}`}
            >
              {optimizeResult.delta_from_start >= 0 ? '↑' : '↓'}{' '}
              {(optimizeResult.delta_from_start * 100).toFixed(1)}% from start
            </span>
          </div>
          <p className={styles.ciText}>
            95% CI: [{(optimizeResult.confidence_interval[0] * 100).toFixed(1)}%,{' '}
            {(optimizeResult.confidence_interval[1] * 100).toFixed(1)}%]
          </p>
          <div className={styles.suggestedList}>
            {PARAM_KEYS.map((key) => {
              const isLocked = lockedParams.has(key)
              const startVal = params[key]
              const nextVal = optimizeResult.suggested_params[key] ?? startVal
              const deltaPct =
                Math.abs(startVal) < 1e-9 ? 0 : ((nextVal - startVal) / startVal) * 100
              return (
                <div key={`opt-${key}`} className={styles.suggestedRow}>
                  <span>{isLocked ? '🔒' : '✓'}</span>
                  <span>{PARAM_LABEL[key]}</span>
                  <span className={styles.valueMono}>{nextVal.toFixed(1)}</span>
                  <span className={styles.suggestedMeta}>
                    {isLocked ? '(locked)' : `${deltaPct >= 0 ? '↑' : '↓'}${Math.abs(deltaPct).toFixed(1)}%`}
                  </span>
                </div>
              )
            })}
          </div>
          <div className={styles.optimizeActions}>
            <button
              type="button"
              className={`${styles.applyBtn} interactive`}
              onClick={() => {
                setParams((prev) => ({
                  ...prev,
                  ...optimizeResult.suggested_params,
                }))
              }}
            >
              Apply to sliders
            </button>
            <button
              type="button"
              className={`${styles.dismissBtn} interactive`}
              onClick={() => setOptimizeResult(null)}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {activeTab === 'optimize' && (
        <div className={styles.optimizeHead}>
          {zeroLocks ? (
            <div className={styles.optimizeWarn}>
              ⚠ Lock at least one variable to constrain the optimization. Without constraints,
              the optimizer will always return the same global optimum regardless of your
              settings.
            </div>
          ) : (
            <p className={styles.optimizeCount}>
              {lockCount} locked · {freeCount} free to optimize
            </p>
          )}
          {inlineWarning && <p className={styles.inlineWarningText}>{inlineWarning}</p>}
        </div>
      )}

      <div className={`${styles.sliderRows} ${tabTransitionClass}`}>
        {PARAM_KEYS.map((key) => {
        const [lo, hi] = DEFAULT_BOUNDS[key]
        const isLocked = lockedParams.has(key)
        const lockIcon = isLocked ? '🔒' : '🔓'
        const optimizeLockedDisable = activeTab === 'optimize' && isLocked
        return (
          <div
            key={key}
            className={`${styles.field} ${isLocked ? styles.fieldLocked : ''} ${activeTab === 'optimize' && !isLocked ? styles.fieldOptimizeFree : ''}`}
          >
            <div className={styles.fieldLabel}>
              <div className={styles.labelLeft}>
                <button
                  type="button"
                  className={`${styles.lockBtn} ${isLocked ? styles.lockBtnOn : ''}`}
                  onClick={() => toggleLock(key)}
                  aria-label={`${isLocked ? 'Unlock' : 'Lock'} ${PARAM_LABEL[key]}`}
                >
                  {lockIcon}
                </button>
                <span className={styles.labelText}>{PARAM_LABEL[key]}</span>
                {activeTab === 'optimize' && isLocked && (
                  <span className={styles.fixedBadge}>FIXED: {params[key].toFixed(1)}</span>
                )}
                {activeTab === 'optimize' && !isLocked && (
                  <span className={styles.startBadge}>
                    start: {params[key].toFixed(1)}{' '}
                    <em className={styles.willOptimize}>will optimize</em>
                  </span>
                )}
              </div>
              <span className={`${styles.valueMono} ${isLocked ? styles.valueLocked : ''}`}>
                <input
                  type="text"
                  inputMode="decimal"
                  className={`${styles.valueInput} ${isLocked ? styles.valueInputLocked : ''}`}
                  value={
                    valueDrafts[key] !== undefined
                      ? valueDrafts[key]
                      : params[key].toFixed(2)
                  }
                  onFocus={() =>
                    setValueDrafts((prev) => ({
                      ...prev,
                      [key]: params[key].toString(),
                    }))
                  }
                  onChange={(e) =>
                    setValueDrafts((prev) => ({
                      ...prev,
                      [key]: e.target.value,
                    }))
                  }
                  onBlur={() => commitParamValue(key)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      commitParamValue(key)
                      ;(e.currentTarget as HTMLInputElement).blur()
                    } else if (e.key === 'Escape') {
                      setValueDrafts((prev) => {
                        const next = { ...prev }
                        delete next[key]
                        return next
                      })
                      ;(e.currentTarget as HTMLInputElement).blur()
                    }
                  }}
                  disabled={optimizeLockedDisable}
                  aria-label={`${PARAM_LABEL[key]} value`}
                />
              </span>
            </div>
            <input
              type="range"
              min={lo}
              max={hi}
              step={(hi - lo) / 200}
              value={params[key]}
              onChange={(e) =>
                setParams((p) => ({
                  ...p,
                  [key]: clamp(Number(e.target.value), lo, hi),
                }))
              }
              className={`${styles.slider} interactive ${isLocked ? styles.sliderLocked : ''}`}
              disabled={optimizeLockedDisable}
            />
          </div>
        )
        })}
      </div>

      {activeTab === 'predict' ? (
        <button
          type="button"
          className={`${styles.predictBtn} interactive`}
          onClick={onPredict}
          disabled={busy}
        >
          {busy ? 'Running model…' : 'Predict yield'}
        </button>
      ) : (
        <>
          <button
            type="button"
            className={`${styles.optimizeBtn} ${zeroLocks ? styles.optimizeBtnDisabled : ''} ${isOptimizing ? styles.optimizeBtnBusy : ''} interactive`}
            onClick={onOptimize}
            aria-disabled={zeroLocks}
          >
            {isOptimizing ? 'Optimizing...' : '⚡ Optimize Free Variables'}
          </button>
          {isOptimizing && optimizeProgress && (
            <p className={styles.optimizeProgress}>
              Evaluating parameter space... ({optimizeProgress.current}/
              {optimizeProgress.total} samples)
            </p>
          )}
        </>
      )}

      {err && <p className={styles.err}>{err.message}</p>}

      {data && (
        <div className={styles.result}>
          <p className={styles.yieldBig}>
            {(data.predicted_yield * 100).toFixed(1)}% die yield
          </p>

          <div className={styles.ciBar}>
            <div
              className={styles.ciFill}
              style={{
                left: `${clamp(data.ci_low, 0, 1) * 100}%`,
                width: `${clamp(data.ci_high - data.ci_low, 0, 1) * 100}%`,
              }}
            />
          </div>
          <p className={styles.ciText}>
            95% CI {(data.ci_low * 100).toFixed(2)}% —{' '}
            {(data.ci_high * 100).toFixed(2)}%
          </p>

          <div className={styles.badgeRow}>
            <span
              className={`${styles.pfBadge} ${data.predicted_yield >= 0.9 ? styles.pfPass : styles.pfFail}`}
            >
              {data.predicted_yield >= 0.9 ? 'PASS TARGET' : 'BELOW TARGET'}
            </span>
          </div>
          {data.spc_alerts_active && (
            <div className={styles.badgeRow}>
              <span
                className={`${styles.spcBadge} ${data.spc_severity < 1.5 ? styles.spcWarn : styles.spcCrit}`}
              >
                ⚠ SPC Alert — {data.spc_alert_count} parameter(s) out of control
              </span>
            </div>
          )}

          {data.risk_flags.length > 0 && (
            <div className={styles.riskRow}>
              {data.risk_flags.map((f) => (
                <span key={f} className={riskClass(f)}>
                  {f}
                </span>
              ))}
            </div>
          )}

          {shapBars.length > 0 && (
            <>
              <p className={styles.shapTitle}>SHAP contribution</p>
              <div className={styles.shapChart}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={shapBars}
                    layout="vertical"
                    margin={{ left: 4, right: 8 }}
                  >
                    <CartesianGrid {...chartGridProps} />
                    <XAxis
                      type="number"
                      stroke="var(--border)"
                      tick={chartTickProps}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={100}
                      stroke="var(--border)"
                      tick={{
                        fill: 'var(--text-muted)',
                        fontSize: 11,
                        fontFamily: 'var(--font-sans)',
                      }}
                    />
                    <Tooltip contentStyle={chartTooltipContentStyle} />
                    <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                      {shapBars.map((entry) => (
                        <Cell
                          key={entry.name}
                          fill={
                            entry.value >= 0
                              ? 'var(--accent-teal)'
                              : 'var(--accent-red)'
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
