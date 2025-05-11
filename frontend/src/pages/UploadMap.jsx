import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import DashboardHeader from '../components/DashboardHeader';
import axios from 'axios';
import { config } from '../config';

const API_BASE_URL = config.API_BASE_URL;

function UploadMap() {
  const navigate = useNavigate();
  const [iframeCode, setIframeCode] = useState('');
  const [editorLink, setEditorLink] = useState('');
  const [error, setError] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState({ type: '', text: '' });
  const [isInstructionsOpen, setIsInstructionsOpen] = useState(false);

  useEffect(() => {
    const fetchSavedIframe = async () => {
      try {
        const token = localStorage.getItem('accessToken');
        const response = await axios.get(`${API_BASE_URL}/api/map/`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        
        if (response.data) {
          if (response.data.iframe_code) {
            setIframeCode(response.data.iframe_code);
          }
          if (response.data.editor_link) {
            setEditorLink(response.data.editor_link);
          }
        }
      } catch (error) {
        console.error('Error fetching saved iframe:', error);
      }
    };

    fetchSavedIframe();
  }, []);

  const validateIframeCode = () => {
    if (!iframeCode.trim()) {
      setError('Please enter an iframe code');
      return false;
    }

    if (!iframeCode.includes('<iframe') || !iframeCode.includes('</iframe>')) {
      setError('Please enter a valid iframe code');
      return false;
    }

    return true;
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
              Upload Map <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">Preview</span>
            </h1>
          </div>

          <div className="bg-zinc-900 rounded-xl p-6 mb-8 shadow-lg">
            <form className="space-y-6">
              <div>
                <label htmlFor="iframeCode" className="block text-sm font-medium text-gray-300 mb-2">
                  Iframe Code
                </label>
                <div className="bg-zinc-800/50 text-gray-400 text-sm p-3 rounded-lg mb-2">
                  <button 
                    type="button" // Add this to prevent form submission
                    onClick={(e) => {
                      e.preventDefault(); // Add this to prevent default behavior
                      setIsInstructionsOpen(!isInstructionsOpen);
                    }}
                    className="flex items-center justify-between w-full text-left font-medium"
                  >
                    <span>How to get the embed code</span>
                    <svg 
                      className={`w-5 h-5 transition-transform duration-200 ${isInstructionsOpen ? 'rotate-180' : ''}`}
                      fill="none" 
                      viewBox="0 0 24 24" 
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  
                  {isInstructionsOpen && (
                    <ol className="ml-4 space-y-1 mt-2 animate-fadeIn">
                      <li>1. Open your map in Mappedin</li>
                      <li>2. Click <span className="text-purple-400">Preview</span></li>
                      <li>3. Click <span className="text-purple-400">Share</span></li>
                      <li>4. Select <span className="text-purple-400">Live</span> tab</li>
                      <li>5. Copy the <span className="text-purple-400">embed code</span></li>
                    </ol>
                  )}
                </div>
                <textarea
                  id="iframeCode"
                  value={iframeCode}
                  onChange={(e) => setIframeCode(e.target.value)}
                  className="w-full h-32 bg-zinc-800 text-white rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="Paste your iframe code here..."
                />
              </div>
              

              {error && (
                <div className="text-red-500 text-sm">
                  {error}
                </div>
              )}

              <div className="flex flex-col space-y-3">

                <button
                  type="button"
                  onClick={async () => {
                    try {
                      setIsSaving(true);
                      setSaveMessage({ type: '', text: '' });
                      
                      const token = localStorage.getItem('accessToken');
                      await axios.post(`${API_BASE_URL}/api/map/`, 
                        { 
                          iframe_code: iframeCode,
                          editor_link: editorLink
                        },
                        { headers: { Authorization: `Bearer ${token}` }}
                      );
                      
                      setSaveMessage({ type: 'success', text: 'Map saved successfully!' });
                    } catch (error) {
                      setSaveMessage({ 
                        type: 'error', 
                        text: error.response?.data?.message || 'Error saving map'
                      });
                    } finally {
                      setIsSaving(false);
                    }
                  }}
                  disabled={!iframeCode.trim() || isSaving}
                  className={`w-full py-2 px-4 bg-gradient-to-r from-blue-700 to-purple-700 text-white rounded-lg hover:from-blue-600 hover:to-purple-600 transition-colors duration-200 transform hover:scale-105 ${(!iframeCode.trim() || isSaving) ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {isSaving ? 'Saving...' : 'Save Map'}
                </button>
                {saveMessage.text && (
                  <div className={`text-sm ${saveMessage.type === 'success' ? 'text-green-400' : 'text-red-400'}`}>
                    {saveMessage.text}
                  </div>
                )}
              </div>
            </form>
          </div>


        </div>
      </div>
    </div>
  );
}

export default UploadMap;