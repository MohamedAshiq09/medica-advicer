export default function Footer() {
  return (
    <footer className="bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 mt-16">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              About MediCheck AI
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              An AI-powered medical triage system that helps assess symptoms and provides 
              guidance on appropriate care levels. Built with machine learning and medical expertise.
            </p>
          </div>

          {/* Important Notice */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Important Notice
            </h3>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
              <li>• This is not medical advice</li>
              <li>• Always consult healthcare professionals</li>
              <li>• For emergencies, call emergency services</li>
              <li>• Educational and research purposes only</li>
            </ul>
          </div>

          {/* Technical Info */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Technology
            </h3>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
              <li>• Machine Learning Algorithms</li>
              <li>• Natural Language Processing</li>
              <li>• Association Rule Mining</li>
              <li>• Multi-layer Safety System</li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-200 dark:border-gray-700 mt-8 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              © 2024 MediCheck AI. Educational project for medical symptom assessment.
            </p>
            <div className="flex items-center space-x-4 mt-4 md:mt-0">
              <span className="text-xs text-gray-500 dark:text-gray-500">
                Backend: FastAPI + ML | Frontend: Next.js + TypeScript
              </span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}