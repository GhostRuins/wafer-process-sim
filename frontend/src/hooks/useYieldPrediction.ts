import { useMutation } from '@tanstack/react-query'

import { apiUrl } from '../config'

export type ToolChoice = 'A' | 'B' | 'C'

export const TOOL_ID_FROM_CHOICE: Record<ToolChoice, string> = {
  A: 'tool_A',
  B: 'tool_B',
  C: 'tool_C',
}

export interface PredictRequest {
  temperature: number
  pressure: number
  gas_flow: number
  rf_power: number
  deposition_time: number
  tool_id: string
}

export interface ShapEntry {
  feature: string
  value: number
}

export interface PredictResponse {
  predicted_yield: number
  ci_low: number
  ci_high: number
  shap_values: ShapEntry[]
  /** Optional SHAP baseline (E[f(x)]) for waterfall start */
  expected_value?: number
  risk_flags: string[]
  spc_alerts_active: boolean
  spc_severity: number
  spc_alert_count: number
  parameter_bounds?: Partial<
    Record<
      | 'temperature'
      | 'pressure'
      | 'gas_flow'
      | 'rf_power'
      | 'deposition_time',
      [number, number]
    >
  >
}

interface ApiYieldPrediction {
  predicted_yield: number
  confidence_interval: [number, number]
  shap_breakdown: Record<string, number>
  risk_flags: string[]
  spc_alerts_active?: boolean
  spc_severity?: number
  spc_alert_count?: number
}

async function postPredict(body: PredictRequest): Promise<PredictResponse> {
  const res = await fetch(apiUrl('/api/predict'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Predict failed: ${res.status}`)
  }
  const json = (await res.json()) as ApiYieldPrediction
  const shap_values = Object.entries(json.shap_breakdown ?? {}).map(
    ([feature, value]) => ({ feature, value }),
  )
  const [lo, hi] = json.confidence_interval ?? [0, 0]
  return {
    predicted_yield: json.predicted_yield,
    ci_low: lo,
    ci_high: hi,
    shap_values,
    risk_flags: json.risk_flags ?? [],
    spc_alerts_active: Boolean(json.spc_alerts_active),
    spc_severity: Number(json.spc_severity ?? 0),
    spc_alert_count: Number(json.spc_alert_count ?? 0),
  }
}

export function useYieldPrediction() {
  return useMutation({
    mutationKey: ['predict-yield'],
    mutationFn: postPredict,
  })
}
