import type { CSSProperties } from 'react'

/** Shared Recharts styling — uses CSS variables from index.css */

export const chartGridProps = {
  strokeDasharray: '3 3' as const,
  stroke: 'var(--border)',
  strokeWidth: 0.5,
}

export const chartTickProps = {
  fill: 'var(--text-muted)',
  fontSize: 11,
}

export const chartTooltipContentStyle: CSSProperties = {
  background: 'var(--bg-elevated)',
  border: '1px solid var(--border)',
  borderRadius: '6px',
  fontSize: '12px',
  color: 'var(--text-primary)',
  fontFamily: 'var(--font-mono)',
}
