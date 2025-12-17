export default function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 mt-16">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="text-lg font-light text-black mb-4">
              About MediCheck AI
            </h3>
            <p className="text-sm text-gray-600 font-light">
              An AI-powered medical triage system that helps assess symptoms and provides 
              guidance on appropriate care levels. Built with machine learning and medical expertise.
            </p>
          </div>

          {/* Important Notice */}
          <div>
            <h3 className="text-lg font-light text-black mb-4">
              Important Notice
            </h3>
            <ul className="text-sm text-gray-600 space-y-2 font-light">
              <li>• This is not medical advice</li>
              <li>• Always consult healthcare professionals</li>
              <li>• For emergencies, call emergency services</li>
              <li>• Educational and research purposes only</li>
            </ul>
          </div>

          {/* Technical Info */}
          <div>
            <h3 className="text-lg font-light text-black mb-4">
              Technology
            </h3>
            <ul className="text-sm text-gray-600 space-y-2 font-light">
              <li>• Machine Learning Algorithms</li>
              <li>• Natural Language Processing</li>
              <li>• Association Rule Mining</li>
              <li>• Multi-layer Safety System</li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-200 mt-8 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-gray-600 font-light">
              © 2024 MediCheck AI. Educational project for medical symptom assessment.
            </p>
            <div className="flex items-center space-x-4 mt-4 md:mt-0">
              <span className="text-xs text-gray-500 font-light">
                Backend: FastAPI + ML | Frontend: Next.js + TypeScript
              </span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}