import React, { useState, useEffect } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import DashboardHeader from '../components/DashboardHeader';
import axios from 'axios';
import { Line, Bar } from 'react-chartjs-2';
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

function ShopliftingDetection() {
  const [cameras, setCameras] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    videoPath: ''
  });
  const [isDeleting, setIsDeleting] = useState({});
  const [deletingCameraIds, setDeletingCameraIds] = useState([]);
  const navigate = useNavigate();
  const location = useLocation();
  const isAnalysisPage = location.pathname.includes('/analysis');

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
        `${API_BASE_URL}/api/get-cameras/`, 
        headers
      );
      if (response.data.status === 'success') {
        const activeCameras = response.data.cameras.filter(cam => cam.id);
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

  useEffect(() => {
    loadCameras();
    const refreshInterval = setInterval(loadCameras, 2000); // Refresh every 2 seconds
    
    return () => clearInterval(refreshInterval);
  }, [navigate]);

  const handleConnectCamera = async () => {
    const headers = getAuthHeaders();
    if (!headers) return;

    try {
      const cameraId = `camera-${Date.now()}`;
      const response = await axios.post(
        `${API_BASE_URL}/api/connect-camera/`,
        {
          camera_id: cameraId,
          video_path: formData.videoPath,
          name: formData.name
        },
        headers
      );

      if (response.data.status === 'success') {
        loadCameras();
        setShowModal(false);
        setFormData({ name: '', videoPath: '' });
      }
    } catch (error) {
      if (error.response?.status === 401) {
        localStorage.removeItem('accessToken');
        navigate('/login');
      }
      console.error('Error connecting camera:', error);
    }
  };

  const handleDeleteCamera = async (cameraId) => {
    if (isDeleting[cameraId] || deletingCameraIds.includes(cameraId)) return; // Prevent multiple delete attempts
    setDeletingCameraIds(prev => [...prev, cameraId]);
    const headers = getAuthHeaders();
    if (!headers) return;
    try {
      setIsDeleting(prev => ({ ...prev, [cameraId]: true }));
      await axios.delete(
        `${API_BASE_URL}/api/delete-camera/${cameraId}/`,
        headers
      );
      // Poll until camera is gone from backend
      let pollCount = 0;
      const poll = async () => {
        pollCount++;
        const response = await axios.get(`${API_BASE_URL}/api/get-cameras/`, headers);
        const stillThere = response.data.cameras.some(cam => cam.id === cameraId);
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
    if (deletingCameraIds.includes(cameraId)) return;
    const headers = getAuthHeaders();
    if (!headers) return;
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/get-frame/${cameraId}/`, 
        headers
      );
      if (response.data.status === 'success' && response.data.frame) {
        const frameElement = document.getElementById(`frame-${cameraId}`);
        if (frameElement) {
          frameElement.src = `data:image/jpeg;base64,${response.data.frame}`;
          frameElement.dataset.lastUpdated = Date.now();
        }
      }
    } catch (error) {
      if (error.response?.status === 404) {
        // Camera not found, stop polling for this camera
        setCameras(prev => prev.filter(cam => cam.id !== cameraId));
        return;
      }
      if (error.response?.status === 401) {
        localStorage.removeItem('accessToken');
        navigate('/login');
      }
      console.error(`Error getting frame for camera ${cameraId}:`, error);
    }
  };

  // Function to check if frames are stuck and restart polling if needed
  const checkStuckFrames = (cameraId) => {
    const frameElement = document.getElementById(`frame-${cameraId}`);
    if (frameElement && frameElement.dataset.lastUpdated) {
      const lastUpdated = parseInt(frameElement.dataset.lastUpdated);
      const now = Date.now();
      // If frame hasn't updated in 10 seconds, consider it stuck
      if (now - lastUpdated > 10000) {
        console.log(`Camera ${cameraId} appears to be stuck, restarting polling`);
        // Reset the image to indicate a connection issue
        frameElement.src = '';
        // Force a frame data refresh on next poll
        frameElement.dataset.lastUpdated = now;
      }
    }
  };

  useEffect(() => {
    const intervals = {};
    const stuckCheckerIntervals = {};
    cameras.forEach(camera => {
      if (!deletingCameraIds.includes(camera.id)) {
        intervals[camera.id] = setInterval(() => pollFrames(camera.id), 100);
        stuckCheckerIntervals[camera.id] = setInterval(() => checkStuckFrames(camera.id), 5000);
      }
    });
    return () => {
      Object.values(intervals).forEach(clearInterval);
      Object.values(stuckCheckerIntervals).forEach(clearInterval);
    };
  }, [cameras, deletingCameraIds]);

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

  function AnalysisView() {
    const [timeFilter, setTimeFilter] = useState('1h');
    const [analysisData, setAnalysisData] = useState({
      realtime_stats: {
        total_suspicious_frames: 0,
        current_avg_rate: 0,
        active_cameras: 0
      },
      time_series: [],
      camera_comparison: []
    });

    const fetchAnalysisData = async () => {
      const headers = getAuthHeaders();
      if (!headers) return;

      try {
        const response = await axios.get(
          `${API_BASE_URL}/api/analysis-data/?time_filter=${timeFilter}`,
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

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: 'rgb(156, 163, 175)',
            font: {
              size: 14
            }
          }
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              return `Percentage: ${context.raw.toFixed(1)}%`;
            }
          },
          padding: 10,
          titleFont: {
            size: 14
          },
          bodyFont: {
            size: 14
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Percentage',
            color: 'rgb(156, 163, 175)',
            font: {
              size: 14,
              weight: 'bold'
            }
          },
          ticks: { 
            color: 'rgb(156, 163, 175)',
            callback: (value) => `${value}%`,
            font: {
              size: 12
            },
            maxTicksLimit: 8  // Limit number of y-axis ticks
          },
          grid: { 
            color: 'rgba(31, 41, 55, 0.2)',
            drawBorder: false
          }
        },
        x: {
          title: {
            display: true,
            text: 'Time',
            color: 'rgb(156, 163, 175)',
            font: {
              size: 14,
              weight: 'bold'
            }
          },
          ticks: { 
            color: 'rgb(156, 163, 175)',
            maxRotation: 45,
            minRotation: 45,
            font: {
              size: 12
            },
            maxTicksLimit: timeFilter === '1h' ? 12 : 12,  // Show all 5-minute intervals for 1h view
            autoSkip: timeFilter !== '1h',  // Don't skip labels for 1h view
            autoSkipPadding: 20
          },
          grid: { 
            color: 'rgba(31, 41, 55, 0.2)',
            drawBorder: false
          }
        }
      },
      elements: {
        line: {
          tension: 0.4,
          borderWidth: 3,  // Increased line thickness
          fill: 'start',
          backgroundColor: 'rgba(147, 51, 234, 0.1)'
        },
        point: {
          radius: 4,  // Slightly larger points
          hitRadius: 10,
          hoverRadius: 6,
          borderWidth: 2  // Added border width for points
        }
      },
      interaction: {
        intersect: false,
        mode: 'index'
      }
    };

    const lineChartData = {
      labels: analysisData.time_series.map(entry => {
        const date = new Date(entry.timestamp);
        if (timeFilter === '1h') {
          return date.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit'
          });
        } else if (timeFilter === '7d') {
          return date.toLocaleDateString([], { 
            weekday: 'short',
            month: 'short', 
            day: 'numeric',
            hour: '2-digit'
          });
        } else if (timeFilter === '30d') {
          return date.toLocaleDateString([], { 
            month: 'short', 
            day: 'numeric'
          });
        } else {
          // 1 year view - show month and year
          return date.toLocaleDateString([], { 
            month: 'short',
            year: 'numeric'
          });
        }
      }),
      datasets: [{
        label: 'Average Shoplifting Percentage',
        data: analysisData.time_series.map(entry => entry.percentage),
        borderColor: 'rgb(147, 51, 234)',
        backgroundColor: 'rgba(147, 51, 234, 0.5)',
        pointBackgroundColor: 'rgb(147, 51, 234)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgb(147, 51, 234)',
        tension: 0.4
      }]
    };

    const barChartData = {
      labels: analysisData.camera_comparison.map(entry => entry.camera_name),
      datasets: [{
        label: 'Shoplifting Event Distribution',
        data: analysisData.camera_comparison.map(entry => entry.percentage),
        backgroundColor: 'rgba(147, 51, 234, 0.7)',
        borderColor: 'rgb(147, 51, 234)',
        borderWidth: 1
      }]
    };

    return (
      <div className="space-y-6">
        {/* Time Filter */}
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
            <h3 className="text-gray-400 text-sm font-medium">Total Suspicious Frames</h3>
            <div className="mt-2 flex items-baseline">
              <p className="text-4xl font-semibold text-purple-500">
                {analysisData.realtime_stats.total_suspicious_frames}
              </p>
              <p className="ml-2 text-sm text-gray-400">last 10s</p>
            </div>
          </div>
          
          <div className="bg-zinc-900 p-6 rounded-lg">
            <h3 className="text-gray-400 text-sm font-medium">Average Suspicion Rate</h3>
            <div className="mt-2 flex items-baseline">
              <p className="text-4xl font-semibold text-purple-500">
                {analysisData.realtime_stats.current_avg_rate.toFixed(1)}%
              </p>
              <p className="ml-2 text-sm text-gray-400">all cameras</p>
            </div>
          </div>
          
          <div className="bg-zinc-900 p-6 rounded-lg">
            <h3 className="text-gray-400 text-sm font-medium">Active Cameras</h3>
            <div className="mt-2 flex items-baseline">
              <p className="text-4xl font-semibold text-purple-500">
                {analysisData.realtime_stats.active_cameras}
              </p>
              <p className="ml-2 text-sm text-gray-400">monitoring</p>
            </div>
          </div>
        </div>
  
        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-zinc-900 p-6 rounded-lg">
            <h3 className="text-white text-lg font-medium mb-4">Average Shoplifting Percentage Over Time</h3>
            <div className="h-80">
              <Line data={lineChartData} options={chartOptions} />
            </div>
          </div>
  
          <div className="bg-zinc-900 p-6 rounded-lg">
            <h3 className="text-white text-lg font-medium mb-4">Camera Zone Comparison</h3>
            <div className="h-80">
              <Bar data={barChartData} options={chartOptions} />
            </div>
          </div>
        </div>
      </div>
    );
  }

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
            Shoplifting <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">Detection</span>
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
            to="/shoplifting-detection"
            className={`pb-2 px-1 ${!isAnalysisPage 
              ? 'text-purple-500 border-b-2 border-purple-500' 
              : 'text-gray-400 hover:text-gray-300'}`}
          >
            Camera
          </Link>
          <Link
            to="/shoplifting-detection/analysis"
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
          // Camera View
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {cameras.map(camera => (
              <div key={camera.id} className="bg-zinc-900 rounded-lg p-4 relative">
                {/* Spinner overlay while deleting */}
                {deletingCameraIds.includes(camera.id) && (
                  <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center z-10">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
                  </div>
                )}
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-white">{camera.name || `Camera ${camera.id}`}</h3>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => { console.log('Delete clicked', camera.id); handleDeleteCamera(camera.id); }}
                      className="text-gray-400 hover:text-white"
                    >
                      ×
                    </button>
                  </div>
                </div>
                <div className="aspect-video bg-black relative overflow-hidden rounded-lg">
                  <img
                    id={`frame-${camera.id}`}
                    className="absolute inset-0 w-full h-full object-contain"
                    alt="Video feed"
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Add Camera Modal */}
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

export default ShopliftingDetection;