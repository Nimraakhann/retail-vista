import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import DashboardHeader from '../components/DashboardHeader';
import AnimatedBackground from '../components/AnimatedBackground';
import '../styles/animations.css';

function Dashboard() {
  const navigate = useNavigate();
  const location = useLocation();
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editorLink, setEditorLink] = useState('');
  const [iframeCode, setIframeCode] = useState('');

  useEffect(() => {
    // Simple auth check
    const accessToken = localStorage.getItem('accessToken');
    const userDataStr = localStorage.getItem('userData');

    if (!accessToken || !userDataStr) {
      navigate('/login');
      return;
    }

    try {
      const userData = JSON.parse(userDataStr);
      setUserData(userData);
    } catch (error) {
      console.error('Error parsing user data:', error);
      navigate('/login');
    } finally {
      setLoading(false);
    }

    // Fetch editor link and iframe code
    const fetchMapData = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/map/', {
          headers: { Authorization: `Bearer ${accessToken}` }
        });
        if (response.data?.editor_link) {
          setEditorLink(response.data.editor_link);
        }
        if (response.data?.iframe_code) {
          setIframeCode(response.data.iframe_code);
        }
      } catch (error) {
        console.error('Error fetching map data:', error);
      }
    };

    fetchMapData();
  }, [navigate]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  if (!userData) {
    return null;
  }

  return (
    <div className="relative w-full overflow-x-hidden bg-black min-h-screen">
      <DashboardHeader />
      <AnimatedBackground />

      {/* Main content */}
      <div className="relative z-10">
        <div className="max-w-7xl mx-auto px-6 py-16 md:px-8">
          {/* Back button */}
          {location.state?.from && (
            <button
              onClick={() => navigate(-1)}
              className="group flex items-center space-x-2 text-white/70 hover:text-purple-400 transition-all duration-300 mb-8 perspective hover:scale-105"
            >
              <svg className="w-5 h-5 transform group-hover:-translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              <span>Back</span>
            </button>
          )}

          {/* Welcome section with 3D effect */}
          <div className="text-center mb-20 relative perspective">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse-slow" />
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-8 relative transform hover:scale-105 transition-transform duration-500">
              Welcome,{' '}
              <span className="bg-gradient-to-r from-purple-400 to-blue-500 text-transparent bg-clip-text animate-gradient">
                {userData.first_name} {userData.last_name}
              </span>
            </h1>
            <p className="text-gray-400 text-xl transform hover:scale-105 transition-transform">
              Manage your profile{' '}
              <button 
                onClick={() => navigate('/profile')}
                className="text-purple-400 hover:text-purple-300 underline focus:outline-none transition-colors"
              >
                here
              </button>
            </p>
          </div>

          {/* Grid layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Store Monitoring */}
            <div className="group relative">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl blur opacity-20 group-hover:opacity-50 transition duration-500"></div>
              <div className="bg-zinc-900 backdrop-blur-sm rounded-xl p-8 border border-zinc-800/50 relative transition-all duration-300 group-hover:scale-[1.02] group-hover:border-purple-500/30">
                <h2 className="text-2xl font-bold text-white mb-6 flex items-center group-hover:text-purple-400 transition-colors">
                  <svg className="w-6 h-6 mr-3 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  Store Monitoring
                </h2>
                <div className="flex flex-col space-y-4">
                  <button 
                    onClick={() => navigate('/shoplifting-detection')}
                    className="w-full py-3 px-6 bg-gradient-to-r from-purple-700 to-blue-700 text-white rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-lg hover:shadow-purple-500/25"
                  >
                    Shoplifting Detection
                  </button>
                  <button 
                    onClick={() => navigate('/age-gender')}
                    className="w-full py-3 px-6 bg-gradient-to-r from-purple-700 to-blue-700 text-white rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-lg hover:shadow-purple-500/25"
                  >
                    Age & Gender Detection
                  </button>
                  <button 
                    onClick={() => navigate('/people-counter')}
                    className="w-full py-3 px-6 bg-gradient-to-r from-purple-700 to-blue-700 text-white rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-lg hover:shadow-purple-500/25"
                  >
                    People Counter
                  </button>
                </div>
              </div>
            </div>

            {/* Map Management */}
            <div className="group relative">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl blur opacity-20 group-hover:opacity-50 transition duration-500"></div>
              <div className="bg-zinc-900 backdrop-blur-sm rounded-xl p-8 border border-zinc-800/50 relative transition-all duration-300 group-hover:scale-[1.02] group-hover:border-purple-500/30">
                <h2 className="text-2xl font-bold text-white mb-6 flex items-center group-hover:text-purple-400 transition-colors">
                  <svg className="w-6 h-6 mr-3 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                  Map Management
                </h2>
                <div className="flex flex-col space-y-4">
                  <button 
                    onClick={() => navigate('/edit-map')}
                    className="w-full py-3 px-6 bg-gradient-to-r from-purple-700 to-blue-700 text-white rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-lg hover:shadow-purple-500/25"
                  >
                    Create/Edit Map
                  </button>
                  <button 
                    onClick={() => navigate('/upload-map')}
                    className="w-full py-3 px-6 bg-gradient-to-r from-purple-700 to-blue-700 text-white rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-lg hover:shadow-purple-500/25"
                  >
                    Upload Map
                  </button>
                  <button 
                    onClick={() => window.open(window.location.origin + '/show-map', '_blank')}
                    className="w-full py-3 px-6 bg-gradient-to-r from-purple-700 to-blue-700 text-white rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-lg hover:shadow-purple-500/25"
                  >
                    Show Map
                  </button>
                </div>
              </div>
            </div>

            {/* Activity & Promotions */}
            <div className="group relative">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl blur opacity-20 group-hover:opacity-50 transition duration-500"></div>
              <div className="bg-zinc-900 backdrop-blur-sm rounded-xl p-8 border border-zinc-800/50 relative transition-all duration-300 group-hover:scale-[1.02] group-hover:border-purple-500/30 h-full flex flex-col">
                <h2 className="text-2xl font-bold text-white mb-6 flex items-center group-hover:text-purple-400 transition-colors">
                  <svg className="w-6 h-6 mr-3 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Recent Activity
                </h2>
                <div className="flex flex-col space-y-4 flex-grow justify-center">
                  <button 
                    onClick={() => navigate('/recent-activity')}
                    className="w-full py-3 px-6 bg-gradient-to-r from-purple-700 to-blue-700 text-white rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-lg hover:shadow-purple-500/25"
                  >
                    View Recent Activity
                  </button>
                </div>
              </div>
            </div>

            {/* Promotions Management */}
            <div className="group relative">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl blur opacity-20 group-hover:opacity-50 transition duration-500"></div>
              <div className="bg-zinc-900 backdrop-blur-sm rounded-xl p-8 border border-zinc-800/50 relative transition-all duration-300 group-hover:scale-[1.02] group-hover:border-purple-500/30 h-full flex flex-col">
                <h2 className="text-2xl font-bold text-white mb-6 flex items-center group-hover:text-purple-400 transition-colors">
                  <svg className="w-6 h-6 mr-3 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                  </svg>
                  Promotions Management
                </h2>
                <div className="flex flex-col space-y-4 flex-grow">
                  <button 
                    onClick={() => navigate('/manage-promotions')}
                    className="w-full py-3 px-6 bg-gradient-to-r from-purple-700 to-blue-700 text-white rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-lg hover:shadow-purple-500/25"
                  >
                    Manage Promotions
                  </button>
                  <button 
                    onClick={() => navigate('/display-promotions')}
                    className="w-full py-3 px-6 bg-gradient-to-r from-purple-700 to-blue-700 text-white rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-lg hover:shadow-purple-500/25"
                  >
                    Display Promotions
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
