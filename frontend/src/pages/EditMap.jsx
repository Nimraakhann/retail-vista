import React from 'react';
import { useNavigate } from 'react-router-dom';
import DashboardHeader from '../components/DashboardHeader';

function EditMap() {
  const navigate = useNavigate();

  const handleOpenEditor = () => {
    window.open(import.meta.env.VITE_MAPPEDIN_EDITOR_URL || 'https://app.mappedin.com/editor/', '_blank', 'noopener,noreferrer');
  };

  return (
    <div className="flex flex-col min-h-screen bg-black">
      <DashboardHeader />
      
      <div className="flex-1 p-4 md:p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center mb-8">
            <button
              onClick={() => navigate(-1)}
              className="text-white hover:text-purple-500 transition-colors mr-4"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </button>

            <h1 className="text-3xl font-bold text-white">
              Map <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">Editor</span>
            </h1>
          </div>

          <div className="bg-zinc-900 rounded-xl p-6 shadow-lg">
            <div className="flex flex-col items-center justify-center space-y-6 py-12">
              <p className="text-gray-400 text-center max-w-lg">
                Click the button below to open the MappedinIn Editor in a new tab. Once you've created your map, return here to upload it.
              </p>
              
              <button
                onClick={handleOpenEditor}
                className="px-8 py-4 bg-gradient-to-r from-purple-700 to-blue-700 text-white rounded-lg hover:from-purple-600 hover:to-blue-600 transition-colors duration-200 transform hover:scale-105 flex items-center space-x-2"
              >
                <span>Open Map Editor</span>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </button>

              <button
                onClick={() => navigate('/upload-map')}
                className="text-purple-500 hover:text-purple-400 underline focus:outline-none"
              >
                Upload Map Preview
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default EditMap;