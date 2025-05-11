import React, { useState, useEffect } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import DashboardHeader from '../components/DashboardHeader';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { config } from '../config';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE_URL = config.API_BASE_URL;

function AnalysisView() {
  const [timeFilter, setTimeFilter] = useState('1h');
  const [analysisData, setAnalysisData] = useState({
    realtime_stats: {
      most_common_age: 'N/A',
      gender_ratio: 'N/A',
      active_cameras: 0
    },
    time_series: {
      labels: [],
      male: [],
      female: []
    },
    age_distribution: [0, 0, 0, 0, 0, 0, 0, 0],
    gender_distribution: [0, 0]
  });
  
  const navigate = useNavigate();
  
  const getAuthHeaders = () => {
    const token = localStorage.getItem('accessToken');
    if (!token) {
      navigate('/login');
      return null;
    }
    return {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    };
  };

  const fetchAnalysisData = async () => {
    const headers = getAuthHeaders();
    if (!headers) return;

    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/age-gender-analysis/?time_filter=${timeFilter}`,
        headers
      );
      if (response.data.status === 'success') {
        setAnalysisData(response.data.data);
      }
    } catch (error) {
      if (error.response?.status === 401) {
        localStorage.removeItem('accessToken');
        navigate('/login');
      }
      console.error('Error fetching analysis data:', error);
    }
  };

  useEffect(() => {
    fetchAnalysisData();
    const interval = setInterval(fetchAnalysisData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [timeFilter]);

  // Updated time data with real data from API
  const timeData = {
    labels: analysisData.time_series.labels,
    datasets: [
      {
        label: 'Male',
        data: analysisData.time_series.male,
        borderColor: 'rgb(37, 99, 235)',
        backgroundColor: 'rgba(37, 99, 235, 0.5)',
      },
      {
        label: 'Female',
        data: analysisData.time_series.female,
        borderColor: 'rgb(236, 72, 153)',
        backgroundColor: 'rgba(236, 72, 153, 0.5)',
      }
    ],
  };

  // Updated age data with real data
  const ageData = {
    labels: ['0-3', '4-7', '8-12', '13-20', '21-32', '33-43', '44-53', '60-100'],
    datasets: [{
      label: 'Age Distribution',
      data: analysisData.age_distribution,
      backgroundColor: 'rgba(147, 51, 234, 0.5)',
    }],
  };

  // Updated gender data with real data
  const genderData = {
    labels: ['Male', 'Female'],
    datasets: [{
      label: 'Gender Distribution',
      data: analysisData.gender_distribution,
      backgroundColor: [
        'rgba(37, 99, 235, 0.5)',
        'rgba(236, 72, 153, 0.5)',
      ],
    }],
  };

  return (
    <div className="space-y-8">
      {/* Add Time Filter Buttons */}
      <div className="flex justify-end space-x-2">
        <button
          onClick={() => setTimeFilter('1h')}
          className={`px-4 py-2 rounded ${
            timeFilter === '1h'
              ? 'bg-purple-600 text-white'
              : 'bg-zinc-800 text-gray-400 hover:text-white'
          }`}
        >
          1 Hour
        </button>
        <button
          onClick={() => setTimeFilter('7d')}
          className={`px-4 py-2 rounded ${
            timeFilter === '7d'
              ? 'bg-purple-600 text-white'
              : 'bg-zinc-800 text-gray-400 hover:text-white'
          }`}
        >
          7 Days
        </button>
        <button
          onClick={() => setTimeFilter('30d')}
          className={`px-4 py-2 rounded ${
            timeFilter === '30d'
              ? 'bg-purple-600 text-white'
              : 'bg-zinc-800 text-gray-400 hover:text-white'
          }`}
        >
          30 Days
        </button>
        <button
          onClick={() => setTimeFilter('1y')}
          className={`px-4 py-2 rounded ${
            timeFilter === '1y'
              ? 'bg-purple-600 text-white'
              : 'bg-zinc-800 text-gray-400 hover:text-white'
          }`}
        >
          1 Year
        </button>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-gray-400 text-sm font-medium">Most Common Age Group</h3>
          <div className="mt-2 flex items-baseline">
            <p className="text-4xl font-semibold text-purple-500">{analysisData.realtime_stats.most_common_age}</p>
            <p className="ml-2 text-sm text-gray-400">years</p>
          </div>
        </div>
        
        <div className="bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-gray-400 text-sm font-medium">Gender Ratio (M:F)</h3>
          <div className="mt-2 flex items-baseline">
            <p className="text-4xl font-semibold text-purple-500">{analysisData.realtime_stats.gender_ratio}</p>
            <p className="ml-2 text-sm text-gray-400">ratio</p>
          </div>
        </div>

        <div className="bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-gray-400 text-sm font-medium">Active Cameras</h3>
          <div className="mt-2 flex items-baseline">
            <p className="text-4xl font-semibold text-purple-500">{analysisData.realtime_stats.active_cameras}</p>
            <p className="ml-2 text-sm text-gray-400">online</p>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-white text-lg font-medium mb-4">Age Distribution</h3>
          <Bar 
            data={ageData}
            options={{
              responsive: true,
              scales: {
                y: {
                  beginAtZero: true,
                  grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                  },
                  ticks: { color: 'rgba(255, 255, 255, 0.8)' }
                },
                x: {
                  grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                  },
                  ticks: { color: 'rgba(255, 255, 255, 0.8)' }
                }
              },
              plugins: {
                legend: {
                  labels: { color: 'rgba(255, 255, 255, 0.8)' }
                }
              }
            }}
          />
        </div>

        <div className="bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-white text-lg font-medium mb-4">Gender Distribution</h3>
          <Bar 
            data={genderData}
            options={{
              responsive: true,
              scales: {
                y: {
                  beginAtZero: true,
                  grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                  },
                  ticks: { color: 'rgba(255, 255, 255, 0.8)' }
                },
                x: {
                  grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                  },
                  ticks: { color: 'rgba(255, 255, 255, 0.8)' }
                }
              },
              plugins: {
                legend: {
                  labels: { color: 'rgba(255, 255, 255, 0.8)' }
                }
              }
            }}
          />
        </div>

        <div className="col-span-1 lg:col-span-2 bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-white text-lg font-medium mb-4">Gender Distribution Over Time</h3>
          <Line 
            data={timeData}
            options={{
              responsive: true,
              scales: {
                y: {
                  beginAtZero: true,
                  grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                  },
                  ticks: { color: 'rgba(255, 255, 255, 0.8)' }
                },
                x: {
                  grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                  },
                  ticks: { color: 'rgba(255, 255, 255, 0.8)' }
                }
              },
              plugins: {
                legend: {
                  labels: { color: 'rgba(255, 255, 255, 0.8)' }
                }
              }
            }}
          />
        </div>
      </div>
    </div>
  );
}

function AgeGenderDetection() {
  const [cameras, setCameras] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    videoPath: ''
  });
  const navigate = useNavigate();
  const location = useLocation();
  const isAnalysisPage = location.pathname.includes('/analysis');
  const [isDeleting, setIsDeleting] = useState({});
  const [deletingCameraIds, setDeletingCameraIds] = useState([]);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('accessToken');
    if (!token) {
      navigate('/login');
      return null;
    }
    return {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    };
  };

  const loadCameras = async () => {
    const headers = getAuthHeaders();
    if (!headers) return;

    try {
      console.log('Loading cameras...');
      const response = await axios.get(
        `${API_BASE_URL}/api/get-age-gender-cameras/`, 
        headers
      );
      if (response.data.status === 'success') {
        const activeCameras = response.data.cameras.filter(cam => cam.camera_id);
        console.log('Loaded cameras:', activeCameras);
        setCameras(activeCameras);
      }
    } catch (error) {
      if (error.response?.status === 401) {
        localStorage.removeItem('accessToken');
        navigate('/login');
      }
      console.error('Error loading cameras:', error);
    }
  };

  const handleConnectCamera = async () => {
    const headers = getAuthHeaders();
    if (!headers) return;

    try {
      const cameraId = `camera-${Date.now()}`;
      const isWebcam = formData.videoPath.toLowerCase().trim() === 'webcam';
      
      const response = await axios.post(
        `${API_BASE_URL}/api/connect-age-gender-camera/`,
        {
          camera_id: cameraId,
          name: formData.name,  // Add this
          video_path: isWebcam ? '0' : formData.videoPath,
          camera_type: 'age_gender'
        },
        headers
      );

      if (response.data.status === 'success') {
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setShowModal(false);
        setFormData({ name: '', videoPath: '' });  // Reset both fields
        
        const camerasResponse = await axios.get(
          `${API_BASE_URL}/api/get-age-gender-cameras/`,
          headers
        );
        
        if (camerasResponse.data.status === 'success') {
          setCameras(camerasResponse.data.cameras);
        }
      } else {
        alert(response.data.message || 'Failed to connect camera');
      }
    } catch (error) {
      console.error('Error connecting camera:', error);
      alert(error.response?.data?.message || 'Error connecting camera');
    }
  };

  const handleDeleteCamera = async (cameraId) => {
    if (isDeleting[cameraId] || deletingCameraIds.includes(cameraId)) return; // Prevent multiple delete attempts
    
    setDeletingCameraIds(prev => [...prev, cameraId]);
    const headers = getAuthHeaders();
    if (!headers) return;

    try {
      setIsDeleting(prev => ({ ...prev, [cameraId]: true }));
      // Optimistically update UI
      setCameras(prev => prev.filter(cam => cam.camera_id !== cameraId));
      await axios.delete(
        `${API_BASE_URL}/api/delete-age-gender-camera/${cameraId}/`,
        headers
      );
      // Poll until camera is gone from backend
      let pollCount = 0;
      const poll = async () => {
        pollCount++;
        const response = await axios.get(`${API_BASE_URL}/api/get-age-gender-cameras/`, headers);
        const stillThere = response.data.cameras.some(cam => cam.camera_id === cameraId);
        if (stillThere && pollCount < 20) {
          setTimeout(poll, 500);
        } else {
          setDeletingCameraIds(prev => prev.filter(id => id !== cameraId));
          setIsDeleting(prev => ({ ...prev, [cameraId]: false }));
          loadCameras();
        }
      };
      poll();
    } catch (error) {
      setDeletingCameraIds(prev => prev.filter(id => id !== cameraId));
      setIsDeleting(prev => ({ ...prev, [cameraId]: false }));
      console.error('Error deleting camera:', error);
      if (error.response?.status === 401) {
        localStorage.removeItem('accessToken');
        navigate('/login');
      }
      await loadCameras();
    }
  };

  const pollFrames = async (cameraId) => {
    const headers = getAuthHeaders(); 
    if (!headers) return;
  
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/get-age-gender-frame/${cameraId}/`,
        headers
      );
      
      if (response.data.status === 'success' && response.data.frame) {
        const frameElement = document.getElementById(`frame-${cameraId}`);
        if (frameElement) {
          frameElement.src = `data:image/jpeg;base64,${response.data.frame}`;
        }
      }
    } catch (error) {
      console.error(`Error getting frame for camera ${cameraId}:`, error);
    }
  };
  
  // Use a more reasonable polling frequency
  useEffect(() => {
    const intervals = {};
  
    cameras.forEach(camera => {
      if (camera?.camera_id) {
        intervals[camera.camera_id] = setInterval(() => pollFrames(camera.camera_id), 50); // 20fps
      }
    });
  
    return () => {
      Object.values(intervals).forEach(clearInterval);
    };
  }, [cameras]);

  useEffect(() => {
    // Cleanup function to run when component unmounts
    return () => {
      // Clear all intervals
      cameras.forEach(camera => {
        if (camera?.id) {
          clearInterval(window.pollInterval?.[camera.id]);
          delete window.pollInterval?.[camera.id];
        }
      });
    };
  }, []);

  useEffect(() => {
    loadCameras();
    const refreshInterval = setInterval(loadCameras, 2000); // Refresh every 2 seconds
    
    return () => clearInterval(refreshInterval);
  }, [navigate]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        loadCameras(); // Load immediately when becoming visible
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  return (
    <div className="min-h-screen bg-black">
      <DashboardHeader />
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex items-center mb-6">
          <button
            onClick={() => navigate('/dashboard')} 
            className="text-white hover:text-purple-500 transition-colors mr-4"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <h1 className="text-3xl font-bold text-white">
            Age and Gender <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">Detection</span>
          </h1>
          {!isAnalysisPage && (
            <button
              onClick={() => setShowModal(true)}
              className="ml-auto bg-gradient-to-r from-purple-600 to-blue-600 text-white px-4 py-2 rounded-lg hover:from-purple-700 hover:to-blue-700 transition-colors"
            >
              + Add Camera
            </button>
          )}
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-6 mb-6 border-b border-zinc-800">
          <Link
            to="/age-gender"
            className={`pb-2 px-1 ${!isAnalysisPage 
              ? 'text-purple-500 border-b-2 border-purple-500' 
              : 'text-gray-400 hover:text-gray-300'}`}
          >
            Cameras
          </Link>
          <Link
            to="/age-gender/analysis"
            className={`pb-2 px-1 ${isAnalysisPage 
              ? 'text-purple-500 border-b-2 border-purple-500' 
              : 'text-gray-400 hover:text-gray-300'}`}
          >
            Analysis
          </Link>
        </div>

        {isAnalysisPage ? (
          <AnalysisView />
        ) : (
          // Existing camera grid view
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {cameras.map(camera => (
              <div key={camera.camera_id} className="bg-zinc-900 rounded-lg p-4 relative">
                {/* Spinner overlay while deleting */}
                {deletingCameraIds.includes(camera.camera_id) && (
                  <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center z-10">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
                  </div>
                )}
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-white">
                    {camera.name || `Camera ${camera.camera_id}`}
                  </h3>
                  <button
                    onClick={() => handleDeleteCamera(camera.camera_id)}
                    className="text-gray-400 hover:text-white"
                  >
                    ×
                  </button>
                </div>
                <div className="aspect-video bg-black relative overflow-hidden rounded-lg">
                  <img
                    id={`frame-${camera.camera_id}`}
                    className="absolute inset-0 w-full h-full object-contain"
                    alt="Video feed"
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {showModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-zinc-900 p-6 rounded-lg max-w-md w-full">
              <h2 className="text-2xl font-semibold text-white mb-6 text-center">Connect Camera</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-gray-400 mb-2">Camera Name</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                    placeholder="e.g., Front Door Camera"
                    className="w-full p-3 bg-zinc-800 text-white rounded border border-zinc-700 focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                  />
                </div>
                
                <div>
                  <label className="block text-gray-400 mb-2">Camera Stream Link/IP Address</label>
                  <input
                    type="text"
                    value={formData.videoPath}
                    onChange={(e) => setFormData({...formData, videoPath: e.target.value})}
                    placeholder="e.g., 192.168.1.100 or rtsp://example.com/stream"
                    className="w-full p-3 bg-zinc-800 text-white rounded border border-zinc-700 focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowModal(false);
                    setFormData({ name: '', videoPath: '' });
                  }}
                  className="px-4 py-2 text-gray-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConnectCamera}
                  className="px-6 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
                >
                  Connect
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AgeGenderDetection;
