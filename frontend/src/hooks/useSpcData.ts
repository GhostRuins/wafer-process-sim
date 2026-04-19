import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'

import { API_BASE_URL, apiUrl } from '../config'

export type SpcMetric = 'wafer_yield' | 'mean_film_thickness_nm' | 'mean_defect_density_cm2'
export type SpcMode = 'batch' | 'stream'
export type SpcReplaySpeed = '0.5x' | '1x' | '5x' | '10x' | 'max'
export type SpcStreamState = 'idle' | 'connecting' | 'live' | 'paused' | 'stopped' | 'reconnecting' | 'complete' | 'error' | 'closed' | 'failed'

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
  replaySpeed,
}: {
  start: string
  end: string
  metric: SpcMetric
  toolId?: string
  mode: SpcMode
  replaySpeed: SpcReplaySpeed
}) {
  const [livePoints, setLivePoints] = useState<SpcPoint[]>([])
  const [liveAlerts, setLiveAlerts] = useState<SpcViolation[]>([])
  const [wsStatus, setWsStatus] = useState<SpcStreamState>('idle')
  const [totalRuns, setTotalRuns] = useState(0)
  const [streamFinished, setStreamFinished] = useState(false)
  const [connectToken, setConnectToken] = useState(0)
  const retryRef = useRef(0)
  const retryTimeoutRef = useRef<number | null>(null)
  const streamFinishedRef = useRef(false)
  const wsRef = useRef<WebSocket | null>(null)
  const pendingPointsRef = useRef<SpcPoint[]>([])
  const pendingAlertsRef = useRef<SpcViolation[]>([])
  const flushTimerRef = useRef<number | null>(null)

  const batchQuery = useQuery({
    queryKey: ['spc-batch', start, end, metric, toolId],
    queryFn: () => fetchSpcBatch(start, end, metric, toolId),
    enabled: mode === 'batch',
  })

  useEffect(() => {
    const flush = () => {
      if (pendingPointsRef.current.length > 0) {
        const nextPoints = pendingPointsRef.current
        pendingPointsRef.current = []
        setLivePoints((prev) => [...prev, ...nextPoints].slice(-2000))
      }
      if (pendingAlertsRef.current.length > 0) {
        const nextAlerts = pendingAlertsRef.current
        pendingAlertsRef.current = []
        setLiveAlerts((prev) => [...nextAlerts, ...prev].slice(0, 200))
      }
    }
    flushTimerRef.current = window.setInterval(flush, 100)
    return () => {
      if (flushTimerRef.current !== null) window.clearInterval(flushTimerRef.current)
      flushTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    streamFinishedRef.current = streamFinished
  }, [streamFinished])

  useEffect(() => {
    if (mode !== 'stream') {
      setLivePoints([])
      setLiveAlerts([])
      setWsStatus('idle')
      setTotalRuns(0)
      setStreamFinished(false)
      wsRef.current?.close()
      wsRef.current = null
      return
    }
    let ws: WebSocket | null = null
    let cancelled = false
    const MAX_RETRIES = 3
    const connect = () => {
      if (cancelled) return
      setWsStatus(retryRef.current === 0 ? 'connecting' : 'reconnecting')
      if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
        wsRef.current.close(1000)
      }
      ws = new WebSocket(wsUrl('/ws/spc'))
      wsRef.current = ws
      ws.onopen = () => {
        retryRef.current = 0
        if (retryTimeoutRef.current !== null) {
          window.clearTimeout(retryTimeoutRef.current)
          retryTimeoutRef.current = null
        }
        setWsStatus('live')
        setStreamFinished(false)
        ws?.send(
          JSON.stringify({
            type: 'spc.init',
            metric,
            tool_id: toolId ?? null,
            speed: replaySpeed,
            start,
            end,
          }),
        )
      }
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as {
            metric?: SpcMetric
            series_id?: string
            type?: string
            point?: SpcPoint
            violations?: SpcViolation[]
            total_runs?: number
            total?: number
          }
          if (msg.type === 'spc.connected') {
            if (typeof msg.total_runs === 'number') setTotalRuns(msg.total_runs)
            return
          }
          if (msg.type === 'complete') {
            setStreamFinished(true)
            setWsStatus('complete')
            if (typeof msg.total === 'number' && msg.total > 0) setTotalRuns(msg.total)
            return
          }
          if (msg.metric && msg.metric !== metric) return
          if (toolId && msg.series_id && msg.series_id !== toolId) return
          if (msg.point) pendingPointsRef.current.push(msg.point)
          if (msg.violations?.length) pendingAlertsRef.current.push(...msg.violations)
        } catch {
          // ignore malformed websocket messages
        }
      }
      ws.onclose = (event) => {
        if (cancelled) return
        if (event.code === 1000) {
          setWsStatus('closed')
          return
        }
        if (streamFinishedRef.current) return
        if (retryRef.current < MAX_RETRIES) {
          setWsStatus('reconnecting')
          const delay = Math.min(1000 * 2 ** retryRef.current, 10_000)
          retryRef.current += 1
          retryTimeoutRef.current = window.setTimeout(connect, delay)
        } else {
          setWsStatus('failed')
        }
      }
      ws.onerror = () => {
        setWsStatus('error')
        ws?.close()
      }
    }
    connect()
    return () => {
      cancelled = true
      if (retryTimeoutRef.current !== null) window.clearTimeout(retryTimeoutRef.current)
      retryTimeoutRef.current = null
      ws?.close(1000)
      wsRef.current = null
    }
  }, [mode, metric, toolId, replaySpeed, start, end, connectToken])

  useEffect(() => {
    if (mode !== 'stream') return
    if (wsStatus !== 'live' && wsStatus !== 'paused') return
    const pingTimer = window.setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ command: 'ping' }))
      }
    }, 20_000)
    return () => window.clearInterval(pingTimer)
  }, [mode, wsStatus])

  const points = useMemo(() => (mode === 'stream' ? livePoints : (batchQuery.data?.points ?? [])), [mode, livePoints, batchQuery.data?.points])
  const violations = useMemo(() => (mode === 'stream' ? liveAlerts : (batchQuery.data?.violations ?? [])), [mode, liveAlerts, batchQuery.data?.violations])
  const summary = mode === 'stream' ? null : (batchQuery.data?.summary ?? null)

  const sendCommand = (payload: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload))
    }
  }

  const pauseStream = () => {
    setWsStatus('paused')
    sendCommand({ command: 'pause' })
  }

  const resumeStream = () => {
    setStreamFinished(false)
    setWsStatus('live')
    sendCommand({ command: 'resume' })
  }

  const stopStream = () => {
    setWsStatus('stopped')
    sendCommand({ command: 'stop' })
    wsRef.current?.close(1000)
  }

  const setStreamSpeed = (speed: SpcReplaySpeed) => {
    sendCommand({ command: 'set_speed', speed })
  }

  const resetStream = () => {
    pendingPointsRef.current = []
    pendingAlertsRef.current = []
    setLivePoints([])
    setLiveAlerts([])
    setWsStatus('connecting')
    setTotalRuns(0)
    setStreamFinished(false)
    wsRef.current?.close(1000)
    setConnectToken((v) => v + 1)
  }
  return {
    batchQuery,
    points,
    violations,
    summary,
    wsStatus,
    totalRuns,
    streamFinished,
    resetStream,
    pauseStream,
    resumeStream,
    stopStream,
    setStreamSpeed,
  }
}
