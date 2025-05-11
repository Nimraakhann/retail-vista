import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { config } from '../config';

const API_BASE_URL = config.API_BASE_URL;

function ShowMap() {
  const [iframeCode, setIframeCode] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchMapData = async () => {
      try {
        const token = localStorage.getItem('accessToken');
        const response = await axios.get(`${API_BASE_URL}/api/map/`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        
        if (response.data?.iframe_code) {
          setIframeCode(response.data.iframe_code);
        } else {
          setError('No map has been uploaded yet');
        }
      } catch (error) {
        console.error('Error fetching map data:', error);
        setError('Error loading map');
      } finally {
        setLoading(false);
      }
    };

    fetchMapData();
  }, []);

  const createMarkup = () => {
    return { __html: iframeCode };
  };

  return (
    <div className="flex flex-col min-h-screen bg-black">
      <div className="flex-1 p-4 md:p-8">
        <div className="max-w-7xl mx-auto">
          <div className="bg-zinc-900 rounded-xl p-6 shadow-lg">
            {loading ? (
              <div className="flex justify-center items-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
              </div>
            ) : error ? (
              <div className="text-center py-12">
                <p className="text-red-400 mb-4">{error}</p>
              </div>
            ) : (
              <div className="w-full aspect-video bg-zinc-800 rounded-lg overflow-hidden">
                <div 
                  dangerouslySetInnerHTML={createMarkup()} 
                  className="w-full h-full"
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ShowMap;