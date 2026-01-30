'use client'

import { useState } from 'react'
import { SymptomInput } from '../types/api'
import SymptomForm from './SymptomForm'
import ResultsDisplay from './ResultsDisplay'
import LoadingSpinner from './LoadingSpinner'

export default function SymptomChecker() {
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (formData: SymptomInput) => {
    setIsLoading(true)
    setError(null)
    setResults(null)

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

    try {
      const response = await fetch(`${apiUrl}/check_symptoms`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      console.error('Error checking symptoms:', err)
      setError(
        err instanceof Error 
          ? err.message 
          : 'Failed to connect to the medical service. Please ensure the backend is running on port 8001.'
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setResults(null)
    setError(null)
  }

  return (
    <div className="space-y-6">
      {/* Symptom Input Form */}
      <div className="bg-white border border-gray-200 rounded-3xl shadow-sm p-8">
        <h2 className="text-2xl font-light text-black mb-2">
          Describe Your Symptoms
        </h2>
        <p className="text-gray-600 mb-6 font-light">
          Tell us what you're experiencing and we'll provide AI-powered recommendations
        </p>
        <SymptomForm onSubmit={handleSubmit} isLoading={isLoading} />
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="bg-white border border-gray-200 rounded-3xl shadow-sm p-8">
          <LoadingSpinner />
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-gray-50 border border-gray-300 rounded-3xl p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-gray-700" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-gray-900">
                Connection Error
              </h3>
              <p className="mt-1 text-sm text-gray-700">
                {error}
              </p>
            </div>
          </div>
          <div className="mt-4">
            <button
              onClick={handleReset}
              className="bg-black hover:bg-gray-900 text-white px-4 py-2 rounded-lg text-sm font-light transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      )}

      {/* Results Display */}
      {results && (
        <ResultsDisplay results={results} onReset={handleReset} />
      )}
    </div>
  )
}