import { useMemo } from 'react'

import styles from './DashboardTour.module.css'

export type TourStep = {
  targetId: string
  title: string
  body: string
}

export function DashboardTour({
  open,
  steps,
  stepIndex,
  onNext,
  onSkip,
}: {
  open: boolean
  steps: TourStep[]
  stepIndex: number
  onNext: () => void
  onSkip: () => void
}) {
  const current = steps[stepIndex]
  const rect = useMemo(() => {
    if (!open || !current) return null
    const el = document.getElementById(current.targetId)
    if (!el) return null
    return el.getBoundingClientRect()
  }, [open, current, stepIndex])

  if (!open || !current || !rect) return null

  const cardW = 380
  const top = Math.min(window.innerHeight - 220, rect.bottom + 14)
  const left = Math.min(window.innerWidth - cardW - 16, Math.max(16, rect.left))
  const isLast = stepIndex >= steps.length - 1

  return (
    <div className={styles.overlay} role="dialog" aria-modal>
      <div
        className={styles.highlight}
        style={{
          left: rect.left - 4,
          top: rect.top - 4,
          width: rect.width + 8,
          height: rect.height + 8,
        }}
      />
      <div className={styles.card} style={{ left, top, width: cardW }}>
        <h3 className={styles.title}>{current.title}</h3>
        <p className={styles.body}>{current.body}</p>
        <div className={styles.actions}>
          <button type="button" className={styles.skip} onClick={onSkip}>
            Skip tour
          </button>
          <button type="button" className={styles.next} onClick={onNext}>
            {isLast ? 'Done' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  )
}

