import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DisplayPromotions = () => {
  const [promotions, setPromotions] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);

  // Fetch active promotions
  useEffect(() => {
    const fetchPromotions = async () => {
      try {
        const accessToken = localStorage.getItem('accessToken');
        const response = await axios.get('http://localhost:8000/api/promotions/active/', {
          headers: {
            'Authorization': `Bearer ${accessToken}`
          }
        });
        setPromotions(response.data.promotions.filter(promo => promo.status === 'active'));
        setLoading(false);
      } catch (error) {
        console.error('Error fetching promotions:', error);
        setLoading(false);
      }
    };

    fetchPromotions();
  }, []);

  // Auto-advance carousel every 5 seconds
  useEffect(() => {
    if (promotions.length === 0) return;

    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) => 
        prevIndex === promotions.length - 1 ? 0 : prevIndex + 1
      );
    }, 5000);

    return () => clearInterval(interval);
  }, [promotions]);

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black flex items-center justify-center">
        <div className="text-white text-xl">Loading promotions...</div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black">
      {promotions.map((promotion, index) => (
        <div
          key={promotion.id}
          className={`absolute inset-0 transition-opacity duration-1000 ${
            index === currentIndex ? 'opacity-100' : 'opacity-0'
          }`}
        >
          <img
            src={promotion.image_url}
            alt={promotion.name}
            className="w-full h-full object-contain"
          />
        </div>
      ))}
    </div>
  );
};

export default DisplayPromotions;
