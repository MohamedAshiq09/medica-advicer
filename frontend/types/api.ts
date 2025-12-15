export interface SymptomInput {
  text: string
  age?: number
  gender?: string
}

export interface TriageResponse {
  triage_level: string
  confidence: number
  explanation: string
  detected_symptoms: string[]
  probabilities: {
    [key: string]: number
  }
  applied_overrides: Array<{
    type: string
    reason: string
    [key: string]: any
  }>
  applied_rules: Array<{
    symptoms?: string[]
    action?: string
    confidence?: number
    source?: string
    [key: string]: any
  }>
  follow_up_questions: string[]
  timestamp: string
  model_info: {
    [key: string]: any
  }
}

export interface HealthCheckResponse {
  status: string
  timestamp: string
  model_loaded: boolean
  llm_available: boolean
  apriori_loaded: boolean
}