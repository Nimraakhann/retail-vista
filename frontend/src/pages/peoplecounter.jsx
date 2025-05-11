import React, { useState, useEffect } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import axios from 'axios';
import DashboardHeader from '../components/DashboardHeader';
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [analysisData, setAnalysisData] = useState({
    realtime_stats: {
      most_active_zone: 'N/A',
      most_active_zone_count: 0,
      total_entries: 0,
      least_active_zone: 'N/A',
      least_active_zone_count: 0,
      active_cameras: 0
    },
    zone_traffic_data: {
      labels: [],
      datasets: []
    },
    zone_comparison_data: {
      labels: [],
      datasets: [{
        label: 'Zone Traffic Distribution',
        data: [],
        backgroundColor: []
      }]
    }
  });
  
  const navigate = useNavigate();
  
  const getAuthHeaders = () => {
    const accessToken = localStorage.getItem('accessToken');
    if (!accessToken) {
      navigate('/login');
      return null;
    }
    return {
      headers: { 
        'Authorization': `Bearer ${accessToken}` 
      }
    };
  };

  const fetchAnalysisData = async () => {
    setLoading(true);
    try {
      const headers = getAuthHeaders();
      if (!headers) return;
      
      const response = await axios.get(
        `${API_BASE_URL}/api/people-counter-analysis/?time_filter=${timeFilter}`,
        headers
      );
      
      if (response.data.status === 'success') {
        setAnalysisData(response.data.data);
      } else {
        setError('Failed to load analysis data');
      }
    } catch (error) {
      console.error('Error fetching analysis data:', error);
      setError('Error loading analysis data');
      
      if (error.response?.status === 401) {
        localStorage.removeItem('accessToken');
        navigate('/login');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalysisData();
    
    // Set up polling every 30 seconds
    const interval = setInterval(fetchAnalysisData, 30000);
    
    return () => clearInterval(interval);
  }, [timeFilter]);

  if (loading && !analysisData.zone_traffic_data.labels.length) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center text-red-500 p-4">
        <p>{error}</p>
        <button 
          onClick={fetchAnalysisData} 
          className="mt-2 px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
        >
          Try Again
        </button>
      </div>
    );
  }

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

      {/* Updated Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-gray-400 text-sm font-medium">Most Active Zone</h3>
          <div className="mt-2 flex items-baseline">
            <p className="text-4xl font-semibold text-purple-500">
              {analysisData.realtime_stats.most_active_zone}
            </p>
            <p className="ml-2 text-sm text-gray-400">
              {analysisData.realtime_stats.most_active_zone_count} visitors
            </p>
          </div>
        </div>
        
        <div className="bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-gray-400 text-sm font-medium">Total Zone Entries</h3>
          <div className="mt-2 flex items-baseline">
            <p className="text-4xl font-semibold text-purple-500">
              {analysisData.realtime_stats.total_entries.toLocaleString()}
            </p>
            <p className="ml-2 text-sm text-gray-400">across all zones</p>
          </div>
        </div>

        <div className="bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-gray-400 text-sm font-medium">Least Active Zone</h3>
          <div className="mt-2 flex items-baseline">
            <p className="text-4xl font-semibold text-purple-500">
              {analysisData.realtime_stats.least_active_zone}
            </p>
            <p className="ml-2 text-sm text-gray-400">
              {analysisData.realtime_stats.least_active_zone_count} visitors
            </p>
          </div>
        </div>
      </div>

      {/* Updated Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-white text-lg font-medium mb-4">Zone Traffic Over Time</h3>
          <Line 
            data={analysisData.zone_traffic_data}
            options={{
              responsive: true,
              scales: {
                y: {
                  beginAtZero: true,
                  title: {
                    display: true,
                    text: 'Number of Visitors',
                    color: 'rgba(255, 255, 255, 0.8)'
                  },
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
                },
                title: {
                  display: true,
                  text: 'Zone Traffic Trends',
                  color: 'rgba(255, 255, 255, 0.8)'
                }
              }
            }}
          />
        </div>

        <div className="bg-zinc-900 p-6 rounded-lg">
          <h3 className="text-white text-lg font-medium mb-4">Zone Traffic Distribution</h3>
          <Bar 
            data={analysisData.zone_comparison_data}
            options={{
              responsive: true,
              scales: {
                y: {
                  beginAtZero: true,
                  title: {
                    display: true,
                    text: 'Total Visitors',
                    color: 'rgba(255, 255, 255, 0.8)'
                  },
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
                  display: false
                },
                title: {
                  display: true,
                  text: 'Visitors per Zone',
                  color: 'rgba(255, 255, 255, 0.8)'
                }
              }
            }}
          />
        </div>
      </div>
    </div>
  );
}

function PeopleCounter() {
  const navigate = useNavigate();
  const [cameras, setCameras] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [showPolygonModal, setShowPolygonModal] = useState(false);
  const [formData, setFormData] = useState({  
    name: '',
    videoPath: ''
  });
  const [firstFrame, setFirstFrame] = useState('');
  const [entryPoints, setEntryPoints] = useState([]);
  const [exitPoints, setExitPoints] = useState([]);
  const [isDrawingEntry, setIsDrawingEntry] = useState(true);
  const [tempPoints, setTempPoints] = useState([]);
  const location = useLocation();
  const isAnalysisPage = location.pathname.includes('/analysis');
  const [isDeleting, setIsDeleting] = useState({});
  const [deletingCameraIds, setDeletingCameraIds] = useState([]);
  const [frameError, setFrameError] = useState({});

  const getAuthHeaders = () => {
    const accessToken = localStorage.getItem('accessToken');
    if (!accessToken) {
      navigate('/login');
      return null;
    }
    return {
      headers: { 
        'Authorization': `Bearer ${accessToken}` 
      }
    };
  };

  const loadCameras = async () => {
    const headers = getAuthHeaders();
    if (!headers) return;

    try {
      console.log('Loading cameras...');
      const response = await axios.get(
        `${API_BASE_URL}/api/get-people-counter-cameras/`, 
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

  const handleConnectCamera = async () => {
    const headers = getAuthHeaders();
    if (!headers) return;

    if (!formData.videoPath.trim()) {
        alert('Please enter a valid video path');
        return;
    }

    try {
        const cameraId = `camera-${Date.now()}`;
        const response = await axios.post(
            `${API_BASE_URL}/api/connect-people-counter/`,
            {
                camera_id: cameraId,
                video_path: formData.videoPath
            },
            headers 
        );

        if (response.data.status === 'success') {
            await loadCameras(); // Wait for cameras to load
            setShowModal(false);
            setFormData({ name: '', videoPath: '' });
        } else {
            alert(response.data.message || 'Failed to connect camera');
        }
    } catch (error) {
        console.error('Error connecting camera:', error);
        alert(error.response?.data?.message || 'Failed to connect camera');
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
        `${API_BASE_URL}/api/delete-people-counter-camera/${cameraId}/`,
        headers
      );
      // Poll until camera is gone from backend
      let pollCount = 0;
      const poll = async () => {
        pollCount++;
        const response = await axios.get(`${API_BASE_URL}/api/get-people-counter-cameras/`, headers);
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
        `${API_BASE_URL}/api/get-people-counter-frame/${cameraId}/`,
        headers
      );
      if (response.data.status === 'success' && response.data.frame) {
        const frameElement = document.getElementById(`frame-${cameraId}`);
        if (frameElement) {
          frameElement.src = `data:image/jpeg;base64,${response.data.frame}`;
        }
        setFrameError(prev => ({ ...prev, [cameraId]: false }));
      }
    } catch (error) {
      if (error.response?.status === 404) {
        setFrameError(prev => ({ ...prev, [cameraId]: true }));
        return;
      }
      console.error(`Error getting frame for camera ${cameraId}:`, error);
    }
  };

  const handleCanvasClick = (e) => {
    const canvas = e.target;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    
    const newPoints = [...tempPoints, {x, y}];
    setTempPoints(newPoints);

    // Draw point on canvas
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = isDrawingEntry ? '#00FF00' : '#FF0000';
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI); // Increased point size from 3 to 8
    ctx.fill();
    
    // Draw connecting lines
    if (newPoints.length > 1) {
        ctx.strokeStyle = isDrawingEntry ? '#00FF00' : '#FF0000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(newPoints[newPoints.length - 2].x, newPoints[newPoints.length - 2].y);
        ctx.lineTo(x, y);
        ctx.stroke();
        
        // Close the polygon if 4 points
        if (newPoints.length === 4) {
            ctx.beginPath();
            ctx.moveTo(newPoints[3].x, newPoints[3].y);
            ctx.lineTo(newPoints[0].x, newPoints[0].y);
            ctx.stroke();
        }
    }

    // When 4 points are collected
    if (newPoints.length === 4) {
        if (isDrawingEntry) {
            setEntryPoints(newPoints);
            setIsDrawingEntry(false);
            setTempPoints([]);
        } else {
            setExitPoints(newPoints);
            handleConnectWithPolygons(newPoints);
        }
    }
  };

  const handleConnectWithPolygons = async (exitPoints) => {
    if (!showPolygonModal) return;
  
    try {
      const cameraId = `camera-${Date.now()}`;
      const headers = getAuthHeaders();
      if (!headers) return;
  
      const formattedEntryPoints = entryPoints.map(point => [point.x, point.y]);
      const formattedExitPoints = exitPoints.map(point => [point.x, point.y]);
  
      if (formattedEntryPoints.length !== 4 || formattedExitPoints.length !== 4) {
        alert('Please draw both entry and exit points (4 points each)');
        return;
      }
  
      const response = await axios.post(
        `${API_BASE_URL}/api/connect-people-counter/`,
        {
          camera_id: cameraId,
          name: formData.name,
          video_path: formData.videoPath,
          entry_points: formattedEntryPoints,
          exit_points: formattedExitPoints
        },
        headers
      );
  
      if (response.data.status === 'success') {
        setShowPolygonModal(false);
        await loadCameras();
        setTempPoints([]);
        setEntryPoints([]);
        setExitPoints([]);
        setIsDrawingEntry(true);
        setFormData({ name: '', videoPath: '' });
        setFirstFrame('');
      } else {
        alert(response.data.message || 'Failed to connect camera');
      }
    } catch (error) {
      console.error('Error connecting camera:', error);
      console.error('Error details:', error.response?.data);
      alert('Error connecting to camera. Please check your points and try again.');
    }
  };

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

  useEffect(() => {
    const intervals = {};
    cameras.forEach(camera => {
      if (!deletingCameraIds.includes(camera.id)) {
        intervals[camera.id] = setInterval(() => pollFrames(camera.id), 50);
      }
    });
    return () => {
      Object.values(intervals).forEach(clearInterval);
    };
  }, [cameras, deletingCameraIds]);

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
            People <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">Counter</span>
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
            to="/people-counter"
            className={`pb-2 px-1 ${!isAnalysisPage 
              ? 'text-purple-500 border-b-2 border-purple-500' 
              : 'text-gray-400 hover:text-gray-300'}`}
          >
            Cameras
          </Link>
          <Link
            to="/people-counter/analysis"
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
          // Your existing camera view code
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
                  <h3 className="text-white">
                    {camera.name || `Camera ${camera.id}`}  {/* Update this line */}
                  </h3>
                  <button
                    onClick={() => handleDeleteCamera(camera.id)}
                    className="text-gray-400 hover:text-white"
                  >
                    ×
                  </button>
                </div>
                <div className="aspect-video bg-black relative overflow-hidden rounded-lg">
                  {frameError[camera.id] ? (
                    <div className="flex items-center justify-center h-full w-full text-gray-400">
                      No video available
                    </div>
                  ) : (
                    <img
                      id={`frame-${camera.id}`}
                      className="absolute inset-0 w-full h-full object-contain"
                      alt="Video feed"
                    />
                  )}
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
                  onClick={async () => {
                    const response = await axios.post(
                      `${API_BASE_URL}/api/get-first-frame/`,
                      { video_path: formData.videoPath },
                      getAuthHeaders()
                    );
                    if (response.data.status === 'success') {
                      setFirstFrame(response.data.frame);
                      setShowModal(false);
                      setShowPolygonModal(true);
                    }
                  }}
                  className="px-6 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        )}

        {showPolygonModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-zinc-900 p-6 rounded-lg max-w-4xl w-full relative">
              <button
                onClick={() => {
                  setShowPolygonModal(false);
                  setTempPoints([]);
                  setEntryPoints([]);
                  setExitPoints([]);
                  setIsDrawingEntry(true);
                  setFormData({ name: '', videoPath: '' });
                  setFirstFrame('');
                }}
                className="absolute top-4 right-4 text-white text-2xl font-bold hover:text-gray-400 z-50 w-8 h-8 flex items-center justify-center"
              >
                ×
              </button>
              <h2 className="text-xl text-white mb-2">Draw Entry/Exit Points</h2>
              <p className="text-gray-400 mb-4">
                {isDrawingEntry 
                  ? "Click to draw 4 points for the entry area (green)" 
                  : "Click to draw 4 points for the exit area (red)"
                }
              </p>
              <div className="relative">
                <canvas
                  onClick={handleCanvasClick}
                  className="w-full rounded-lg cursor-crosshair"
                  style={{
                    backgroundImage: `url(data:image/jpeg;base64,${firstFrame})`,
                    backgroundSize: 'contain',
                    backgroundRepeat: 'no-repeat'
                  }}
                  width={1020}
                  height={600}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default PeopleCounter;