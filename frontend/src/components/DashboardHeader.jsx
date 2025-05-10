import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const DashboardHeader = () => {
  const { logout, user } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <header className="bg-white shadow relative z-50">  
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <Link to="/dashboard" className="text-xl font-bold text-gray-800">
              Retail Vista
            </Link>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={handleLogout}
              className="bg-gradient-to-r from-purple-700 to-blue-700 text-white px-6 py-2 rounded-full shadow-lg hover:scale-105 transform transition duration-200"
            >
              Logout
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default DashboardHeader;
