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
        <label htmlFor="symptoms" className="block text-sm font-light text-gray-900 mb-2">
          Describe your symptoms in detail *
        </label>
        <textarea
          id="symptoms"
          rows={4}
          className="w-full px-4 py-3 border border-gray-300 rounded-2xl shadow-sm focus:ring-2 focus:ring-black focus:border-black bg-white text-black placeholder-gray-400 resize-none transition-all font-light"
          placeholder="Example: I have a severe headache, fever, and feel nauseous. The headache started this morning and is getting worse..."
          value={formData.text}
          onChange={handleTextChange}
          required
          disabled={isLoading}
        />
        <p className="mt-2 text-xs text-gray-600 font-light">
          Be as specific as possible about your symptoms, their severity, and duration.
        </p>
      </div>

      {/* Age and Gender Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Age */}
        <div>
          <label htmlFor="age" className="block text-sm font-light text-gray-900 mb-2">
            Age (optional)
          </label>
          <input
            type="number"
            id="age"
            min="0"
            max="120"
            className="w-full px-4 py-3 border border-gray-300 rounded-2xl shadow-sm focus:ring-2 focus:ring-black focus:border-black bg-white text-black placeholder-gray-400 transition-all font-light"
            placeholder="Enter your age"
            value={formData.age || ''}
            onChange={handleAgeChange}
            disabled={isLoading}
          />
        </div>

        {/* Gender */}
        <div>
          <label htmlFor="gender" className="block text-sm font-light text-gray-900 mb-2">
            Gender (optional)
          </label>
          <select
            id="gender"
            className="w-full px-4 py-3 border border-gray-300 rounded-2xl shadow-sm focus:ring-2 focus:ring-black focus:border-black bg-white text-black transition-all font-light"
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
      <div className="flex justify-center pt-4">
        <button
          type="submit"
          disabled={!formData.text.trim() || isLoading}
          className="bg-black hover:bg-gray-900 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-light py-3 px-8 rounded-full transition-all duration-200 flex items-center space-x-2 shadow-sm hover:shadow-md"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing...
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
      <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-2xl">
        <h4 className="text-sm font-light text-gray-900 mb-3">
          Try these sample inputs:
        </h4>
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => setFormData(prev => ({ ...prev, text: "I have a mild headache and feel tired after working long hours" }))}
            className="block w-full text-left text-sm text-gray-700 hover:text-black transition-colors pl-3 py-1 font-light"
            disabled={isLoading}
          >
            • Mild headache and fatigue
          </button>
          <button
            type="button"
            onClick={() => setFormData(prev => ({ ...prev, text: "I have severe chest pain and difficulty breathing that started suddenly" }))}
            className="block w-full text-left text-sm text-gray-700 hover:text-black transition-colors pl-3 py-1 font-light"
            disabled={isLoading}
          >
            • Chest pain with breathing difficulty
          </button>
          <button
            type="button"
            onClick={() => setFormData(prev => ({ ...prev, text: "I have a runny nose, sore throat, and mild fever for 2 days" }))}
            className="block w-full text-left text-sm text-gray-700 hover:text-black transition-colors pl-3 py-1 font-light"
            disabled={isLoading}
          >
            • Cold-like symptoms
          </button>
        </div>
      </div>
    </form>
  )
}