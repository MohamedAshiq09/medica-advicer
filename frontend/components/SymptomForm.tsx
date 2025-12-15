'use client'

import { useState } from 'react'
import { SymptomInput } from '@/types/api'

interface SymptomFormProps {
  onSubmit: (data: SymptomInput) => void
  isLoading: boolean
}

export default function SymptomForm({ onSubmit, isLoading }: SymptomFormProps) {
  const [formData, setFormData] = useState<SymptomInput>({
    text: '',
    age: undefined,
    gender: undefined
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (formData.text.trim()) {
      onSubmit(formData)
    }
  }

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setFormData(prev => ({ ...prev, text: e.target.value }))
  }

  const handleAgeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const age = e.target.value ? parseInt(e.target.value) : undefined
    setFormData(prev => ({ ...prev, age }))
  }

  const handleGenderChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const gender = e.target.value || undefined
    setFormData(prev => ({ ...prev, gender }))
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Symptom Description */}
      <div>
        <label htmlFor="symptoms" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Describe your symptoms in detail *
        </label>
        <textarea
          id="symptoms"
          rows={4}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white resize-none"
          placeholder="Example: I have a severe headache, fever, and feel nauseous. The headache started this morning and is getting worse..."
          value={formData.text}
          onChange={handleTextChange}
          required
          disabled={isLoading}
        />
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          Be as specific as possible about your symptoms, their severity, and duration.
        </p>
      </div>

      {/* Age and Gender Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Age */}
        <div>
          <label htmlFor="age" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Age (optional)
          </label>
          <input
            type="number"
            id="age"
            min="0"
            max="120"
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
            placeholder="Enter your age"
            value={formData.age || ''}
            onChange={handleAgeChange}
            disabled={isLoading}
          />
        </div>

        {/* Gender */}
        <div>
          <label htmlFor="gender" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Gender (optional)
          </label>
          <select
            id="gender"
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
            value={formData.gender || ''}
            onChange={handleGenderChange}
            disabled={isLoading}
          >
            <option value="">Select gender</option>
            <option value="M">Male</option>
            <option value="F">Female</option>
            <option value="Other">Other</option>
          </select>
        </div>
      </div>

      {/* Submit Button */}
      <div className="flex justify-center">
        <button
          type="submit"
          disabled={!formData.text.trim() || isLoading}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium py-3 px-8 rounded-lg transition-colors duration-200 flex items-center space-x-2"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing Symptoms...
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Check Symptoms
            </>
          )}
        </button>
      </div>

      {/* Sample Inputs */}
      <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Try these sample inputs:
        </h4>
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => setFormData(prev => ({ ...prev, text: "I have a mild headache and feel tired after working long hours" }))}
            className="block w-full text-left text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
            disabled={isLoading}
          >
            • Mild headache and fatigue
          </button>
          <button
            type="button"
            onClick={() => setFormData(prev => ({ ...prev, text: "I have severe chest pain and difficulty breathing that started suddenly" }))}
            className="block w-full text-left text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
            disabled={isLoading}
          >
            • Chest pain with breathing difficulty
          </button>
          <button
            type="button"
            onClick={() => setFormData(prev => ({ ...prev, text: "I have a runny nose, sore throat, and mild fever for 2 days" }))}
            className="block w-full text-left text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
            disabled={isLoading}
          >
            • Cold-like symptoms
          </button>
        </div>
      </div>
    </form>
  )
}