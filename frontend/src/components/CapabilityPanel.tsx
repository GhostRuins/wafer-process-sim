import { useMemo, useState } from 'react'

import type { ProcessRun } from '../hooks/useWaferData'
import { DEFAULT_RECIPE_ID, getProcessRecipeSpec } from '../config/processSpecs'
import {
  computeCapabilityRows,
  type CapabilityIndexResult,
  type CapabilityStatus,
} from '../utils/capability'

import styles from './CapabilityPanel.module.css'

export type CapabilityToolScope = 'all' | 'selected'

function formatDateRange(startIso: string, endIso: string): string {
  const a = new Date(startIso)
  const b = new Date(endIso)
  if (Number.isNaN(a.getTime()) || Number.isNaN(b.getTime())) {
    return 'Current dashboard range'
  }
  const opts: Intl.DateTimeFormatOptions = {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }
  return `${a.toLocaleDateString('en-US', opts)} — ${b.toLocaleDateString('en-US', opts)}`
}

function formatIndex(v: number | null): string {
  if (v === null || Number.isNaN(v)) return '—'
  return v.toFixed(2)
}

function formatMean(v: number | null, unit: string): string {
  if (v === null || Number.isNaN(v)) return '—'
  const u = unit === 'degC' ? '°C' : unit
  return `${v.toFixed(2)} ${u}`
}

function statusBadgeClass(status: CapabilityStatus): string {
  switch (status) {
    case 'incapable':
      return styles.badgeIncapable
    case 'marginal':
      return styles.badgeMarginal
    case 'capable':
      return styles.badgeCapable
    default:
      return styles.badgeInsufficient
  }
}

function statusLabel(status: CapabilityStatus): string {
  switch (status) {
    case 'incapable':
      return 'Cpk < 1'
    case 'marginal':
      return '1 ≤ Cpk < 1.33'
    case 'capable':
      return 'Cpk ≥ 1.33'
    default:
      return 'N/A'
  }
}

function cardClassForStatus(status: CapabilityStatus): string {
  switch (status) {
    case 'incapable':
      return styles.cardIncapable
    case 'marginal':
      return styles.cardMarginal
    case 'capable':
      return styles.cardCapable
    default:
      return styles.cardInsufficient
  }
}

function displayUnit(unit: string): string {
  if (unit === 'degC') return '°C'
  return unit
}

function SpecFootnote({ row }: { row: { indices: CapabilityIndexResult } }) {
  const { belowLslCount, aboveUslCount, inSpecCount, sampleCount } = row.indices
  if (sampleCount === 0) return null
  return (
    <p className={styles.paramMeta}>
      In spec: {inSpecCount}/{sampleCount} · below LSL: {belowLslCount} · above USL:{' '}
      {aboveUslCount}
    </p>
  )
}

export function CapabilityPanel({
  runs,
  loading,
  dateStart,
  dateEnd,
  spcToolFilter,
}: {
  runs: ProcessRun[]
  loading: boolean
  dateStart: string
  dateEnd: string
  /** Empty string = all tools in SPC chart filter */
  spcToolFilter: string
}) {
  const [toolScope, setToolScope] = useState<CapabilityToolScope>('all')

  const recipe = useMemo(() => getProcessRecipeSpec(DEFAULT_RECIPE_ID), [])

  const scopedRuns = useMemo(() => {
    if (toolScope === 'all') return runs
    if (!spcToolFilter) return []
    return runs.filter((r) => r.tool_id === spcToolFilter)
  }, [runs, toolScope, spcToolFilter])

  const rows = useMemo(
    () => computeCapabilityRows(scopedRuns, recipe.parameters),
    [scopedRuns, recipe.parameters],
  )

  const rangeLabel = formatDateRange(dateStart, dateEnd)
  const toolLabel =
    toolScope === 'all'
      ? 'All tools (Pp/Ppk — long-term overall)'
      : spcToolFilter
        ? `${spcToolFilter.replace(/^tool_/i, 'Tool ')} (Cp/Cpk — short-term within-run order)`
        : 'Selected tool'

  return (
    <section className={styles.root} aria-label="Process capability indices">
      <div className={styles.head}>
        <div>
          <h4 className={styles.title}>Process capability</h4>
          <p className={styles.subtitle}>
            {recipe.recipeLabel} · {rangeLabel}. Spec limits (LSL/USL) drive Cp/Cpk/Pp/Ppk; SPC
            control limits are separate. σ_within from moving range (n=2 subgroups); σ_overall
            from sample standard deviation.
          </p>
        </div>
      </div>

      <div className={styles.toggleRow}>
        <span className={styles.paramMeta}>Tool scope</span>
        <button
          type="button"
          className={`${styles.toggleBtn} ${toolScope === 'all' ? styles.toggleBtnActive : ''}`}
          onClick={() => setToolScope('all')}
        >
          All tools
        </button>
        <button
          type="button"
          className={`${styles.toggleBtn} ${toolScope === 'selected' ? styles.toggleBtnActive : ''}`}
          onClick={() => setToolScope('selected')}
        >
          SPC tool filter
        </button>
      </div>
      <p className={styles.paramMeta}>{toolLabel}</p>

      {toolScope === 'selected' && !spcToolFilter ? (
        <p className={styles.warn}>
          Choose a specific tool in the SPC chart filter above to evaluate tool-scoped capability.
        </p>
      ) : null}

      {loading ? (
        <p className={styles.empty}>Loading runs for capability…</p>
      ) : scopedRuns.length < 2 ? (
        <p className={styles.empty}>
          Need at least two runs in range for capability indices
          {toolScope === 'selected' && spcToolFilter ? ` (${scopedRuns.length} for this tool).` : '.'}
        </p>
      ) : (
        <div className={styles.grid}>
          {rows.map(({ parameter, indices }) => (
            <article
              key={parameter.key}
              className={`${styles.card} ${cardClassForStatus(indices.status)}`}
            >
              <div>
                <h5 className={styles.paramName}>{parameter.label}</h5>
                <p className={styles.paramMeta}>
                  Spec: LSL {parameter.specLimits.lsl} · USL {parameter.specLimits.usl} · target{' '}
                  {parameter.specLimits.target} {displayUnit(parameter.unit)} · Eng. window{' '}
                  {parameter.engineeringLimits.min}–{parameter.engineeringLimits.max}{' '}
                  {displayUnit(parameter.unit)}
                </p>
                <p className={styles.paramMeta}>
                  μ = {formatMean(indices.mean, parameter.unit)} · σ_within ={' '}
                  {formatIndex(indices.sigmaWithin)} · σ_overall ={' '}
                  {formatIndex(indices.sigmaOverall)} · n = {indices.sampleCount}
                </p>
                <SpecFootnote row={{ indices }} />
              </div>
              <span className={`${styles.badge} ${statusBadgeClass(indices.status)}`}>
                {statusLabel(indices.status)}
              </span>
              <div className={styles.metrics}>
                <div className={styles.metricCell}>
                  <span className={styles.metricLabel}>Cp</span>
                  <span className={styles.metricValue}>{formatIndex(indices.cp)}</span>
                </div>
                <div className={styles.metricCell}>
                  <span className={styles.metricLabel}>Cpk</span>
                  <span className={styles.metricValue}>{formatIndex(indices.cpk)}</span>
                </div>
                <div className={styles.metricCell}>
                  <span className={styles.metricLabel}>Pp</span>
                  <span className={styles.metricValue}>{formatIndex(indices.pp)}</span>
                </div>
                <div className={styles.metricCell}>
                  <span className={styles.metricLabel}>Ppk</span>
                  <span className={styles.metricValue}>{formatIndex(indices.ppk)}</span>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
