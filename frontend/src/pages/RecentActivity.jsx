import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import DashboardHeader from '../components/DashboardHeader';
import axios from 'axios';
import { config } from '../config';
import Pagination from '../components/Pagination';

const API_BASE_URL = config.API_BASE_URL;

function RecentActivity() {
  const [alerts, setAlerts] = useState([]);
  const [filteredAlerts, setFilteredAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [showConfirmReviewModal, setShowConfirmReviewModal] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [sortOrder, setSortOrder] = useState('newest'); // 'newest', 'oldest'
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
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

  const fetchAlerts = async () => {
    setLoading(true);
    const headers = getAuthHeaders();
    if (!headers) return;

    try {
      // Add cache-busting parameter to prevent browser caching
      const timestamp = new Date().getTime();
      const response = await axios.get(
        `${API_BASE_URL}/api/shoplifting-alerts/?t=${timestamp}`,
        headers
      );
      
      if (response.data.status === 'success') {
        setAlerts(response.data.alerts);
        applyFilters(response.data.alerts);
      }
    } catch (error) {
      console.error('Error fetching alerts:', error);
      if (error.response?.status === 401) {
        navigate('/login');
      }
    } finally {
      setLoading(false);
    }
  };

  // Apply sort only
  const applyFilters = (alertData) => {
    let filtered = [...alertData];
    
    // Apply sort
    filtered.sort((a, b) => {
      const dateA = new Date(a.timestamp);
      const dateB = new Date(b.timestamp);
      
      if (sortOrder === 'newest') {
        return dateB - dateA;
      } else {
        return dateA - dateB;
      }
    });
    
    setFilteredAlerts(filtered);
  };

  useEffect(() => {
    fetchAlerts();
    // Refresh data every 30 seconds
    const intervalId = setInterval(fetchAlerts, 30000);
    return () => clearInterval(intervalId);
  }, [navigate]);
  
  // Re-apply filters when sort criteria changes
  useEffect(() => {
    applyFilters(alerts);
  }, [sortOrder]);

  // Calculate paginated alerts
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentAlerts = filteredAlerts.slice(indexOfFirstItem, indexOfLastItem);

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  const handleItemsPerPageChange = (newSize) => {
    setItemsPerPage(newSize);
    setCurrentPage(1);
  };

  const handlePlayVideo = (alert) => {
    setSelectedAlert(alert);
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    // Small delay before clearing to allow animation to complete
    setTimeout(() => {
      setSelectedAlert(null);
    }, 300);
  };

  const handleConfirmReview = (alert) => {
    setSelectedAlert(alert);
    setShowConfirmReviewModal(true);
  };

  const handleMarkAsReviewed = async () => {
    if (!selectedAlert) return;
    
    const headers = getAuthHeaders();
    if (!headers) return;
    
    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/mark-alert-as-reviewed/${selectedAlert.id}/`,
        {},
        headers
      );
      
      if (response.data.status === 'success') {
        // Remove the alert from the UI since it will be deleted on the server
        const updatedAlerts = alerts.filter(alert => alert.id !== selectedAlert.id);
        setAlerts(updatedAlerts);
        
        // Apply filters to the updated alerts list
        applyFilters(updatedAlerts);
        
        // Close the confirm modal first
        setShowConfirmReviewModal(false);
        
        // If this was from the video player, close that too after a short delay
        if (showModal) {
          setTimeout(() => {
            handleCloseModal();
          }, 500);
        }
        
        // Add the alert ID to processed alerts in localStorage to prevent notifications
        try {
          const processedAlerts = JSON.parse(localStorage.getItem('processedAlerts') || '[]');
          processedAlerts.push(selectedAlert.id);
          localStorage.setItem('processedAlerts', JSON.stringify(processedAlerts));
        } catch (e) {
          console.error("Error updating processed alerts in localStorage:", e);
        }
        
        // Show a temporary success message
        const messageElement = document.createElement('div');
        messageElement.className = 'fixed bottom-4 right-4 bg-green-600 text-white py-2 px-4 rounded-lg shadow-lg';
        messageElement.textContent = 'Alert marked as reviewed and scheduled for deletion';
        document.body.appendChild(messageElement);
        
        setTimeout(() => {
          messageElement.style.opacity = '0';
          messageElement.style.transition = 'opacity 0.5s ease-in-out';
          setTimeout(() => {
            document.body.removeChild(messageElement);
          }, 500);
        }, 3000);
      }
    } catch (error) {
      console.error('Error marking alert as reviewed:', error);
      
      // Show error message
      const messageElement = document.createElement('div');
      messageElement.className = 'fixed bottom-4 right-4 bg-red-600 text-white py-2 px-4 rounded-lg shadow-lg';
      messageElement.textContent = 'Error marking alert as reviewed';
      document.body.appendChild(messageElement);
      
      setTimeout(() => {
        messageElement.style.opacity = '0';
        messageElement.style.transition = 'opacity 0.5s ease-in-out';
        setTimeout(() => {
          document.body.removeChild(messageElement);
        }, 500);
      }, 3000);
    } finally {
      setShowConfirmReviewModal(false);
    }
  };

  // Helper to ensure video_clip is an absolute URL
  const getVideoUrl = (video_clip) => {
    if (!video_clip) return null;
    if (/^https?:\/\//i.test(video_clip)) return video_clip;
    // If relative, prepend backend base URL
    return `${API_BASE_URL}${video_clip.startsWith('/') ? '' : '/'}${video_clip}`;
  };

  const handleDownload = (alert) => {
    const videoUrl = getVideoUrl(alert.video_clip);
    if (!videoUrl) return;
    try {
      const link = document.createElement('a');
      link.href = videoUrl;
      if (link.href.indexOf('?') === -1) {
        link.href += `?t=${Date.now()}`;
      } else {
        link.href += `&t=${Date.now()}`;
      }
      link.download = `shoplifting-${alert.camera_name}-${new Date(alert.timestamp).toISOString().replace(/:/g, '-')}.mp4`;
      link.target = '_blank';
      document.body.appendChild(link);
      link.click();
      setTimeout(() => {
        document.body.removeChild(link);
      }, 100);
      console.log("Download initiated for video:", link.href);
    } catch (error) {
      console.error("Error initiating download:", error);
      window.open(videoUrl, '_blank');
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

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
            Recent <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">Activities</span>
          </h1>
          
          <div className="ml-auto flex items-center space-x-2">
            <span className="text-gray-400 text-sm mr-2">
              Videos auto-delete after 15 days
            </span>
            <button
              onClick={() => setSortOrder('newest')}
              className={`px-3 py-2 rounded-l border ${
                sortOrder === 'newest' 
                  ? 'bg-purple-600 border-purple-500 text-white' 
                  : 'bg-zinc-800 border-zinc-700 text-gray-400'
              }`}
            >
              Newest
            </button>
            <button
              onClick={() => setSortOrder('oldest')}
              className={`px-3 py-2 rounded-r border ${
                sortOrder === 'oldest' 
                  ? 'bg-purple-600 border-purple-500 text-white' 
                  : 'bg-zinc-800 border-zinc-700 text-gray-400'
              }`}
            >
              Oldest
            </button>
          </div>
        </div>
        
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="text-white">Loading...</div>
          </div>
        ) : filteredAlerts.length === 0 ? (
          <div className="bg-zinc-900 rounded-lg p-8 text-center">
            <p className="text-gray-400 text-lg">No suspicious activities found.</p>
          </div>
        ) : (
          <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {currentAlerts.map(alert => (
              <div 
                key={alert.id} 
                className="bg-zinc-900 rounded-lg p-4 transition-all duration-300 hover:scale-[1.02] hover:shadow-lg hover:shadow-purple-500/10"
              >
                <div className="relative aspect-video bg-black rounded-lg overflow-hidden mb-3">
                  <img 
                    className="absolute inset-0 w-full h-full object-cover" 
                    src={alert.thumbnail || import.meta.env.VITE_PLACEHOLDER_IMAGE || '/placeholder.jpg'}
                    alt={`${alert.camera_name} alert`}
                  />
                  <button
                    onClick={() => handlePlayVideo(alert)}
                    className="absolute inset-0 w-full h-full flex items-center justify-center"
                  >
                    <div className="bg-purple-600 bg-opacity-70 rounded-full p-4 transform hover:scale-110 transition-transform">
                      <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                      </svg>
                    </div>
                  </button>
                  {alert.is_reviewed && (
                    <div className="absolute top-2 right-2 bg-green-600 text-white text-xs font-bold px-2 py-1 rounded">
                      Reviewed
                    </div>
                  )}
                </div>
                
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="text-white font-medium">{alert.camera_name}</h3>
                    <p className="text-gray-400 text-sm">{formatTimestamp(alert.timestamp)}</p>
                  </div>
                </div>
                
                <div className="flex space-x-2">
                  {!alert.is_reviewed && (
                    <button 
                      onClick={() => handleConfirmReview(alert)}
                      className="flex-1 py-2 px-3 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded transition-colors"
                    >
                      Mark as Reviewed
                    </button>
                  )}
                  
                  <button 
                    onClick={() => handleDownload(alert)}
                    className="flex-1 py-2 px-3 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors"
                  >
                    Download
                  </button>
                </div>
              </div>
            ))}
          </div>
          <div className="[&_select]:bg-purple-700 [&_select]:text-white [&_option]:bg-purple-700 mt-6">
            <Pagination
              totalItems={filteredAlerts.length}
              itemsPerPage={itemsPerPage}
              currentPage={currentPage}
              onPageChange={handlePageChange}
              onItemsPerPageChange={handleItemsPerPageChange}
            />
          </div>
          </>
        )}
      </div>
      
      {/* Video Player Modal */}
      {showModal && selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
          <div className="bg-zinc-900 rounded-lg max-w-4xl w-full overflow-hidden relative">
            <button 
              onClick={handleCloseModal}
              className="absolute top-4 right-4 text-white hover:text-purple-400 transition-colors z-10"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            
            <div className="p-4">
              <h3 className="text-white text-xl font-bold mb-2">
                {selectedAlert.camera_name} - {formatTimestamp(selectedAlert.timestamp)}
              </h3>
            </div>
            
            <div className="aspect-video bg-black">
              {selectedAlert.video_clip &&
                /^https?:\/\/.+\.(mp4|webm)(\?.*)?$/i.test(getVideoUrl(selectedAlert.video_clip)) ? (
                <video 
                  className="w-full h-full object-contain" 
                  src={getVideoUrl(selectedAlert.video_clip)}
                  controls 
                  autoPlay
                  loop
                  playsInline
                  onError={(e) => {
                    console.error("Video error:", e);
                    e.target.poster = selectedAlert.thumbnail || '/placeholder.jpg';
                    e.target.style.display = 'none';
                    const fallback = document.createElement('div');
                    fallback.className = 'w-full h-full flex items-center justify-center text-gray-400';
                    fallback.innerHTML = `<div class='text-center'><p class='mb-2'>Unable to play video</p><p class='text-sm'>You can try downloading the video instead.</p></div>`;
                    e.target.parentNode.appendChild(fallback);
                  }}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-400">
                  <div className="text-center">
                    <p className="mb-2">No video available</p>
                    <p className="text-sm">Thumbnail image is available</p>
                  </div>
                </div>
              )}
            </div>
            
            <div className="p-4 flex justify-end space-x-3">
              {!selectedAlert.is_reviewed && (
                <button
                  onClick={() => handleConfirmReview(selectedAlert)}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm font-medium transition-colors"
                >
                  Mark as Reviewed
                </button>
              )}
              
              <button
                onClick={() => handleDownload(selectedAlert)}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium transition-colors"
              >
                Download Video
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Confirm Review Modal */}
      {showConfirmReviewModal && selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
          <div className="bg-zinc-900 rounded-lg max-w-md w-full p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Confirm Review</h3>
            <p className="text-gray-300 mb-6">
              Are you sure you want to mark this alert as reviewed? 
              This action indicates you've reviewed the incident and the clip will be deleted once marked as reviewed.
            </p>
            
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowConfirmReviewModal(false)}
                className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleMarkAsReviewed}
                className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded transition-colors"
              >
                Yes, Mark as Reviewed
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default RecentActivity; 