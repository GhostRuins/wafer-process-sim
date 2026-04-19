import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'

import { API_BASE_URL, apiUrl } from '../config'

export type SpcMetric = 'wafer_yield' | 'mean_film_thickness_nm' | 'mean_defect_density_cm2'
export type SpcMode = 'batch' | 'stream'

export type SpcPoint = {
  index: number
  timestamp: string
  value: number
  cl: number
  ucl: number
  lcl: number
  ewma: number
  ewma_ucl: number
  ewma_lcl: number
  violation_ids: number[]
}

export type SpcViolation = {
  rule_id: number
  rule_name: string
  severity: 'low' | 'medium' | 'high'
  index: number
  timestamp: string
  value: number
  message: string
  evidence_indices: number[]
}

export type SpcBatchResponse = {
  summary: {
    total_points: number
    total_violations: number
    f1?: number | null
  }
  points: SpcPoint[]
  violations: SpcViolation[]
}

function wsUrl(path: string): string {
  const base = API_BASE_URL.replace(/^http/, 'ws')
  const p = path.startsWith('/') ? path : `/${path}`
  return `${base}${p}`
}

async function fetchSpcBatch(start: string, end: string, metric: SpcMetric, toolId?: string): Promise<SpcBatchResponse> {
  const res = await fetch(apiUrl('/api/spc/analyze'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      metric,
      start,
      end,
      tool_id: toolId || null,
      include_points: true,
      mode: 'batch',
    }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `SPC request failed: ${res.status}`)
  }
  return (await res.json()) as SpcBatchResponse
}

export function useSpcData({
  start,
  end,
  metric,
  toolId,
  mode,
}: {
  start: string
  end: string
  metric: SpcMetric
  toolId?: string
  mode: SpcMode
}) {
  const [livePoints, setLivePoints] = useState<SpcPoint[]>([])
  const [liveAlerts, setLiveAlerts] = useState<SpcViolation[]>([])
  const [wsStatus, setWsStatus] = useState<'idle' | 'connecting' | 'live' | 'reconnecting'>('idle')
  const retryRef = useRef(0)

  const batchQuery = useQuery({
    queryKey: ['spc-batch', start, end, metric, toolId],
    queryFn: () => fetchSpcBatch(start, end, metric, toolId),
    enabled: mode === 'batch',
  })

  useEffect(() => {
    if (mode !== 'stream') {
      setLivePoints([])
      setLiveAlerts([])
      setWsStatus('idle')
      return
    }
    let ws: WebSocket | null = null
    let cancelled = false
    let timer: number | undefined
    const connect = () => {
      if (cancelled) return
      setWsStatus(retryRef.current === 0 ? 'connecting' : 'reconnecting')
      ws = new WebSocket(wsUrl('/api/spc/stream/ws'))
      ws.onopen = () => {
        retryRef.current = 0
        setWsStatus('live')
      }
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as {
            metric?: SpcMetric
            series_id?: string
            point?: SpcPoint
            violations?: SpcViolation[]
          }
          if (msg.metric && msg.metric !== metric) return
          if (toolId && msg.series_id && msg.series_id !== toolId) return
          if (msg.point) setLivePoints((prev) => [...prev.slice(-999), msg.point!])
          if (msg.violations?.length) setLiveAlerts((prev) => [...msg.violations!, ...prev].slice(0, 200))
        } catch {
          // ignore malformed websocket messages
        }
      }
      ws.onclose = () => {
        if (cancelled) return
        retryRef.current += 1
        const delay = Math.min(10_000, 1000 * retryRef.current)
        timer = window.setTimeout(connect, delay)
      }
      ws.onerror = () => ws?.close()
    }
    connect()
    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
      ws?.close()
    }
  }, [mode, metric, toolId])

  const points = useMemo(() => (mode === 'stream' ? livePoints : (batchQuery.data?.points ?? [])), [mode, livePoints, batchQuery.data?.points])
  const violations = useMemo(() => (mode === 'stream' ? liveAlerts : (batchQuery.data?.violations ?? [])), [mode, liveAlerts, batchQuery.data?.violations])
  const summary = mode === 'stream' ? null : (batchQuery.data?.summary ?? null)
  return { batchQuery, points, violations, summary, wsStatus }
}
