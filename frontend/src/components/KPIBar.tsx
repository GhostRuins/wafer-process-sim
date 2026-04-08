import styles from './KPIBar.module.css'

export interface KPIBarProps {
  avgYieldTodayPct: number | null
  totalRuns: number
  bestTool: string
  worstDieCluster: string
  activeAnomalies: number
  dateStart: string
  dateEnd: string
  onDateStartChange: (v: string) => void
  onDateEndChange: (v: string) => void
  onExportCsv: () => void
  onOpenTour: () => void
  exportDisabled?: boolean
  loading?: boolean
}

export function KPIBar({
  avgYieldTodayPct,
  totalRuns,
  bestTool,
  worstDieCluster,
  activeAnomalies,
  dateStart,
  dateEnd,
  onDateStartChange,
  onDateEndChange,
  onExportCsv,
  onOpenTour,
  exportDisabled,
  loading,
}: KPIBarProps) {
  if (loading) {
    return (
      <header className={styles.skeletonBar}>
        <div className={`${styles.skeletonPulse} skeleton`} />
        <div className={`${styles.skeletonPulse} skeleton`} style={{ maxWidth: 200 }} />
      </header>
    )
  }

  return (
    <header className={styles.bar} id="kpi-bar">
      <div className={styles.kpis}>
        <div className={styles.kpi}>
          <span className={styles.label}>
            AVG DIE YIELD{' '}
            <span
              className={styles.info}
              title="% of dies per wafer passing all process specifications. Predicted pre-electrical-test by the ensemble ML model."
            >
              ⓘ
            </span>
          </span>
          <span className={styles.value}>
            {avgYieldTodayPct === null ? '—' : `${avgYieldTodayPct.toFixed(1)}%`}
          </span>
        </div>
        <div className={styles.kpi}>
          <span className={styles.label}>Total runs</span>
          <span className={styles.value}>{totalRuns}</span>
        </div>
        <div className={styles.kpi}>
          <span className={styles.label}>Best tool</span>
          <span className={`${styles.value} ${styles.valueSm}`}>{bestTool}</span>
        </div>
        <div className={styles.kpi}>
          <span className={styles.label}>Worst die cluster</span>
          <span className={`${styles.value} ${styles.valueSm}`}>
            {worstDieCluster}
          </span>
        </div>
        <div className={styles.kpi}>
          <span className={styles.label}>Active anomalies</span>
          <span className={styles.value}>{activeAnomalies}</span>
        </div>
      </div>
      <div className={styles.actions}>
        <div className={styles.dateField}>
          <span className={styles.dateLabel}>Start</span>
          <input
            type="datetime-local"
            value={dateStart.slice(0, 16)}
            onChange={(e) =>
              onDateStartChange(new Date(e.target.value).toISOString())
            }
            className={styles.dateInput}
          />
        </div>
        <div className={styles.dateField}>
          <span className={styles.dateLabel}>End</span>
          <input
            type="datetime-local"
            value={dateEnd.slice(0, 16)}
            onChange={(e) =>
              onDateEndChange(new Date(e.target.value).toISOString())
            }
            className={styles.dateInput}
          />
        </div>
        <button
          type="button"
          className={`${styles.exportBtn} interactive`}
          onClick={onExportCsv}
          disabled={exportDisabled}
        >
          Export CSV
        </button>
        <button
          type="button"
          className={`${styles.tourBtn} interactive`}
          onClick={onOpenTour}
          aria-label="Open dashboard tour"
        >
          ?
        </button>
      </div>
    </header>
  )
}
