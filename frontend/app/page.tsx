'use client'

import { useState } from 'react'
import SymptomChecker from '@/components/SymptomChecker'
import Header from '@/components/Header'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Medical Symptoms Checker
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
              Get AI-powered triage recommendations based on your symptoms. 
              This tool helps determine if you should seek immediate care, schedule a doctor visit, or manage symptoms at home.
            </p>
            <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
              <p className="text-sm text-yellow-800 dark:text-yellow-200 font-medium">
                ⚠️ Important: This is not medical advice. Always consult healthcare professionals for medical decisions.
              </p>
            </div>
          </div>
          
          <SymptomChecker />
        </div>
      </main>
      <Footer />
    </div>
  )
}