export default function Header() {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {/* Logo */}
            <div className="bg-black p-2 rounded-lg">
              <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19 8h-2v3h-3v2h3v3h2v-3h3v-2h-3V8zM4 6h5v2H4v5h2v2H4v5h5v2H4c-1.1 0-2-.9-2-2V8c0-1.1.9-2 2-2zm0 0V4c0-1.1.9-2 2-2h3v2H6v2H4zm11-2h3c1.1 0 2 .9 2 2v2h-2V4h-3V2z"/>
              </svg>
            </div>
            
            <div>
              <h1 className="text-lg font-light text-black">
                MediCheck AI
              </h1>
              <p className="text-xs text-gray-500">
                Health Assessment
              </p>
            </div>
          </div>

          {/* Status Indicator */}
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-black rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600 font-light">
                Active
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}