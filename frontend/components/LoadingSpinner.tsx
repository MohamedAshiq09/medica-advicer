export default function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center justify-center py-8">
      <div className="relative">
        {/* Outer ring */}
        <div className="w-16 h-16 border-4 border-blue-200 dark:border-blue-800 rounded-full animate-pulse"></div>
        
        {/* Inner spinning ring */}
        <div className="absolute top-0 left-0 w-16 h-16 border-4 border-transparent border-t-blue-600 dark:border-t-blue-400 rounded-full animate-spin"></div>
        
        {/* Medical cross icon in center */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <svg className="w-6 h-6 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 24 24">
            <path d="M19 8h-2v3h-3v2h3v3h2v-3h3v-2h-3V8zM4 6h5v2H4v5h2v2H4v5h5v2H4c-1.1 0-2-.9-2-2V8c0-1.1.9-2 2-2zm0 0V4c0-1.1.9-2 2-2h3v2H6v2H4zm11-2h3c1.1 0 2 .9 2 2v2h-2V4h-3V2z"/>
          </svg>
        </div>
      </div>
      
      <div className="mt-4 text-center">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Analyzing Your Symptoms
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Our AI is processing your information...
        </p>
        
        {/* Progress steps */}
        <div className="mt-4 space-y-2 text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
            <span>Processing symptoms</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
            <span>Running ML analysis</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
            <span>Applying safety checks</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0.6s' }}></div>
            <span>Generating recommendations</span>
          </div>
        </div>
      </div>
    </div>
  )
}