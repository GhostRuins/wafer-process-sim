export type CapabilityParamKey =
  | 'temperature'
  | 'pressure'
  | 'gas_flow'
  | 'rf_power'
  | 'deposition_time'

export interface ProcessParameterSpec {
  key: CapabilityParamKey
  label: string
  unit: string
  engineeringLimits: {
    min: number
    max: number
  }
  specLimits: {
    lsl: number
    usl: number
    target: number
  }
}

export type ProcessRecipeId = 'cvd_standard_v1'

export interface ProcessRecipeSpec {
  recipeId: ProcessRecipeId
  recipeLabel: string
  parameters: Record<CapabilityParamKey, ProcessParameterSpec>
}

export const DEFAULT_RECIPE_ID: ProcessRecipeId = 'cvd_standard_v1'

export const PROCESS_RECIPE_SPECS: Record<ProcessRecipeId, ProcessRecipeSpec> = {
  cvd_standard_v1: {
    recipeId: 'cvd_standard_v1',
    recipeLabel: 'CVD Standard v1',
    parameters: {
      temperature: {
        key: 'temperature',
        label: 'Temperature',
        unit: 'degC',
        engineeringLimits: { min: 320, max: 470 },
        specLimits: { lsl: 370, usl: 430, target: 400 },
      },
      pressure: {
        key: 'pressure',
        label: 'Pressure',
        unit: 'mTorr',
        engineeringLimits: { min: 10, max: 130 },
        specLimits: { lsl: 30, usl: 70, target: 50 },
      },
      gas_flow: {
        key: 'gas_flow',
        label: 'Gas flow',
        unit: 'sccm',
        engineeringLimits: { min: 25, max: 150 },
        specLimits: { lsl: 60, usl: 100, target: 80 },
      },
      rf_power: {
        key: 'rf_power',
        label: 'RF power',
        unit: 'W',
        engineeringLimits: { min: 120, max: 340 },
        specLimits: { lsl: 190, usl: 260, target: 225 },
      },
      deposition_time: {
        key: 'deposition_time',
        label: 'Deposition time',
        unit: 's',
        engineeringLimits: { min: 60, max: 220 },
        specLimits: { lsl: 100, usl: 150, target: 125 },
      },
    },
  },
}

export function getProcessRecipeSpec(
  recipeId: ProcessRecipeId = DEFAULT_RECIPE_ID,
): ProcessRecipeSpec {
  return PROCESS_RECIPE_SPECS[recipeId]
}
