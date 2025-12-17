'use client'

import { useState } from 'react'
import SymptomChecker from '@/components/SymptomChecker'
import Header from '@/components/Header'
import Footer from '@/components/Footer'

export default function Home() {
  const [showChecker, setShowChecker] = useState(false)

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      {!showChecker ? (
        <main className="min-h-[calc(100vh-120px)] flex items-center justify-center px-4 py-20">
          <div className="text-center max-w-2xl mx-auto">
            {/* Animated Icon */}
            <div className="mb-12 flex justify-center">
              <div className="relative w-48 h-48">
                <div className="absolute inset-0 bg-gray-100 rounded-full opacity-40 blur-2xl animate-pulse"></div>
                <svg className="w-full h-full animate-spin" style={{ animationDuration: '10s' }} viewBox="0 0 200 200">
                  <defs>
                    <radialGradient id="sphereGradient" cx="35%" cy="35%">
                      <stop offset="0%" stopColor="#000000" stopOpacity="0.6" />
                      <stop offset="100%" stopColor="#000000" stopOpacity="0.2" />
                    </radialGradient>
                  </defs>
                  <circle cx="100" cy="100" r="90" fill="url(#sphereGradient)" />
                  <circle cx="100" cy="100" r="85" fill="none" stroke="#000000" strokeWidth="0.5" opacity="0.2" />
                  <circle cx="100" cy="100" r="75" fill="none" stroke="#000000" strokeWidth="0.5" opacity="0.15" />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-6 h-6 bg-black rounded-full"></div>
                </div>
              </div>
            </div>

            {/* Hero Text */}
            <h1 className="text-5xl md:text-6xl font-light text-black mb-4 leading-tight tracking-tight">
              MediCheck AI
            </h1>
            <p className="text-lg text-gray-600 mb-8 max-w-xl mx-auto leading-relaxed font-light">
              Smart symptom assessment powered by AI. Get personalized health guidance in seconds.
            </p>

            {/* CTA Button */}
            <button
              onClick={() => setShowChecker(true)}
              className="inline-flex items-center justify-center px-8 py-3 bg-black hover:bg-gray-900 text-white font-light rounded-full shadow-sm hover:shadow-md transition-all duration-300"
            >
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
              </svg>
              Check Symptoms
            </button>

            {/* Disclaimer */}
            <div className="mt-12 p-4 bg-gray-50 rounded-2xl border border-gray-200">
              <p className="text-sm text-gray-600 font-light">
                ⚠️ This is not medical advice. Always consult healthcare professionals for medical decisions.
              </p>
            </div>
          </div>
        </main>
      ) : (
        <main className="container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto">
            <button
              onClick={() => setShowChecker(false)}
              className="mb-8 inline-flex items-center text-gray-600 hover:text-black transition-colors font-light"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Back to Home
            </button>
            <SymptomChecker />
          </div>
        </main>
      )}
      
      <Footer />
    </div>
  )
}