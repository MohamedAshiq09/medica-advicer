'use client'

import { TriageResponse } from '../types/api'

interface ResultsDisplayProps {
  results: TriageResponse
  onReset: () => void
}

export default function ResultsDisplay({ results, onReset }: ResultsDisplayProps) {
  const getTriageColor = (level: string) => {
    switch (level) {
      case 'emergency':
        return 'bg-red-100 border-red-300 text-red-800 dark:bg-red-900/20 dark:border-red-800 dark:text-red-200'
      case 'see_doctor':
        return 'bg-yellow-100 border-yellow-300 text-yellow-800 dark:bg-yellow-900/20 dark:border-yellow-800 dark:text-yellow-200'
      case 'self-care':
        return 'bg-green-100 border-green-300 text-green-800 dark:bg-green-900/20 dark:border-green-800 dark:text-green-200'
      default:
        return 'bg-gray-100 border-gray-300 text-gray-800 dark:bg-gray-900/20 dark:border-gray-800 dark:text-gray-200'
    }
  }

  const getTriageIcon = (level: string) => {
    switch (level) {
      case 'emergency':
        return (
          <svg className="w-6 h-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        )
      case 'see_doctor':
        return (
          <svg className="w-6 h-6 text-yellow-600 dark:text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      case 'self-care':
        return (
          <svg className="w-6 h-6 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      default:
        return null
    }
  }

  const getTriageTitle = (level: string) => {
    switch (level) {
      case 'emergency':
        return 'Seek Emergency Care'
      case 'see_doctor':
        return 'See a Doctor'
      case 'self-care':
        return 'Self-Care Recommended'
      default:
        return 'Assessment Complete'
    }
  }

  const formatConfidence = (confidence: number) => {
    return `${Math.round(confidence * 100)}%`
  }

  return (
    <div className="space-y-6">
      {/* Main Recommendation */}
      <div className={`border rounded-3xl p-6 ${
        results.triage_level === 'emergency' ? 'bg-red-50 border-red-300' :
        results.triage_level === 'see_doctor' ? 'bg-yellow-50 border-yellow-300' :
        'bg-green-50 border-green-300'
      }`}>
        <div className="flex items-center space-x-3 mb-4">
          {getTriageIcon(results.triage_level)}
          <h3 className={`text-xl font-light ${
            results.triage_level === 'emergency' ? 'text-red-900' :
            results.triage_level === 'see_doctor' ? 'text-yellow-900' :
            'text-green-900'
          }`}>
            {getTriageTitle(results.triage_level)}
          </h3>
          <span className={`text-sm font-light px-3 py-1 rounded-full ${
            results.triage_level === 'emergency' ? 'bg-red-200 text-red-900' :
            results.triage_level === 'see_doctor' ? 'bg-yellow-200 text-yellow-900' :
            'bg-green-200 text-green-900'
          }`}>
            {formatConfidence(results.confidence)}
          </span>
        </div>
        
        <div className={`whitespace-pre-line text-sm font-light ${
          results.triage_level === 'emergency' ? 'text-red-800' :
          results.triage_level === 'see_doctor' ? 'text-yellow-800' :
          'text-green-800'
        }`}>
          {results.explanation}
        </div>
      </div>

      {/* Detected Symptoms */}
      {results.detected_symptoms.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-3xl shadow-sm p-6">
          <h4 className="text-lg font-light text-black mb-4">
            Detected Symptoms
          </h4>
          <div className="flex flex-wrap gap-2">
            {results.detected_symptoms.map((symptom, index) => (
              <span
                key={index}
                className="bg-gray-100 text-gray-900 px-3 py-1 rounded-full text-sm font-light border border-gray-300"
              >
                {symptom.replace('_', ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Probability Breakdown */}
      <div className="bg-white border border-gray-200 rounded-3xl shadow-sm p-6">
        <h4 className="text-lg font-light text-black mb-4">
          Assessment Breakdown
        </h4>
        <div className="space-y-3">
          {Object.entries(results.probabilities).map(([level, probability]) => (
            <div key={level} className="flex items-center justify-between">
              <span className="text-sm font-light text-gray-900 capitalize">
                {level.replace('_', ' ').replace('-', ' ')}
              </span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      level === 'emergency' ? 'bg-red-500' :
                      level === 'see_doctor' ? 'bg-yellow-500' :
                      'bg-green-500'
                    }`}
                    style={{ width: `${probability * 100}%` }}
                  />
                </div>
                <span className="text-sm font-light text-gray-700 w-12 text-right">
                  {formatConfidence(probability)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Follow-up Questions */}
      {results.follow_up_questions.length > 0 && (
        <div className="bg-gray-50 border border-gray-200 rounded-3xl p-6">
          <h4 className="text-lg font-light text-black mb-4">
            Additional Questions to Consider
          </h4>
          <ul className="space-y-2">
            {results.follow_up_questions.map((question, index) => (
              <li key={index} className="text-sm text-gray-700 flex items-start font-light">
                <span className="text-gray-900 mr-2">•</span>
                {question}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Applied Rules/Overrides */}
      {(results.applied_overrides.length > 0 || results.applied_rules.length > 0) && (
        <div className="bg-white border border-gray-200 rounded-3xl p-6">
          <h4 className="text-lg font-light text-black mb-4">
            Assessment Details
          </h4>
          
          {results.applied_overrides.length > 0 && (
            <div className="mb-4">
              <h5 className="text-sm font-light text-gray-900 mb-2">
                Safety Overrides Applied:
              </h5>
              {results.applied_overrides.map((override, index) => (
                <div key={index} className="text-sm text-gray-700 bg-gray-50 p-2 rounded-lg mb-2 border border-gray-200 font-light">
                  <span className="font-medium">{override.type}:</span> {override.reason}
                </div>
              ))}
            </div>
          )}

          {results.applied_rules.length > 0 && (
            <div>
              <h5 className="text-sm font-light text-gray-900 mb-2">
                Pattern Rules Applied:
              </h5>
              {results.applied_rules.map((rule, index) => (
                <div key={index} className="text-sm text-gray-700 bg-gray-50 p-2 rounded-lg mb-2 border border-gray-200 font-light">
                  <span className="font-medium">Rule:</span> {rule.symptoms?.join(' + ')} → {rule.action}
                  {rule.confidence && <span className="ml-2">({formatConfidence(rule.confidence)} confidence)</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={onReset}
          className="bg-black hover:bg-gray-900 text-white font-light py-3 px-8 rounded-full transition-all duration-200 shadow-sm hover:shadow-md"
        >
          Check New Symptoms
        </button>
      </div>

      {/* Timestamp */}
      <div className="text-center text-xs text-gray-600 font-light">
        Assessment completed at {new Date(results.timestamp).toLocaleString()}
      </div>
    </div>
  )
}