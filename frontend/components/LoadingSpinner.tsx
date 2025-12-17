export default function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="relative">
        {/* Outer ring */}
        <div className="w-20 h-20 border-4 border-gray-300 rounded-full animate-pulse"></div>
        
        {/* Inner spinning ring */}
        <div className="absolute top-0 left-0 w-20 h-20 border-4 border-transparent border-t-black border-r-black rounded-full animate-spin"></div>
        
        {/* Medical cross icon in center */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <svg className="w-8 h-8 text-black animate-pulse" fill="currentColor" viewBox="0 0 24 24">
            <path d="M19 8h-2v3h-3v2h3v3h2v-3h3v-2h-3V8zM4 6h5v2H4v5h2v2H4v5h5v2H4c-1.1 0-2-.9-2-2V8c0-1.1.9-2 2-2zm0 0V4c0-1.1.9-2 2-2h3v2H6v2H4zm11-2h3c1.1 0 2 .9 2 2v2h-2V4h-3V2z"/>
          </svg>
        </div>
      </div>
      
      <div className="mt-6 text-center">
        <h3 className="text-lg font-light text-black">
          Analyzing Your Symptoms
        </h3>
        <p className="text-sm text-gray-600 mt-2 font-light">
          Our AI is processing your information...
        </p>
        
        {/* Progress steps */}
        <div className="mt-6 space-y-2 text-xs text-gray-600 font-light">
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 bg-black rounded-full animate-pulse"></div>
            <span>Processing symptoms</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 bg-black rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
            <span>Running ML analysis</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 bg-black rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
            <span>Applying safety checks</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 bg-black rounded-full animate-pulse" style={{ animationDelay: '0.6s' }}></div>
            <span>Generating recommendations</span>
          </div>
        </div>
      </div>
    </div>
  )
}