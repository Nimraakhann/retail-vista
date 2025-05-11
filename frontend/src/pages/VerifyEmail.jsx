import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { config } from '../config';

const API_BASE_URL = config.API_BASE_URL;

function VerifyEmail() {
  const [status, setStatus] = useState('verifying');
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const queryParams = new URLSearchParams(location.search);
    const token = queryParams.get('token');

    if (!token) {
      setStatus('error');
      return;
    }

    const verifyEmail = async () => {
      try {
        const response = await axios.post(`${API_BASE_URL}/api/verify-email/`, { token });
        
        if (response.data.status === 'success') {
          setStatus('success');
          // Redirect to login page after 3 seconds
          setTimeout(() => {
            navigate('/login');
          }, 3000);
        } else {
          setStatus('error');
        }
      } catch (error) {
        setStatus('error');
      }
    };

    verifyEmail();
  }, [location.search, navigate]);

  const renderContent = () => {
    switch (status) {
      case 'verifying':
        return (
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-4">Verifying your email...</h2>
            <p>Please wait while we verify your email address.</p>
          </div>
        );
      case 'success':
        return (
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-4 text-green-600">Email Verified!</h2>
            <p>Your email has been successfully verified. You will be redirected to the login page shortly.</p>
          </div>
        );
      case 'error':
        return (
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-4 text-red-600">Verification Failed</h2>
            <p>We couldn't verify your email. The link might be invalid or expired.</p>
            <button
              onClick={() => navigate('/signup')}
              className="mt-4 px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
            >
              Back to Sign Up
            </button>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      
      <div className="flex-grow flex items-center justify-center">
        <div className="max-w-md w-full mx-4">
          <div className="bg-white p-8 rounded-lg shadow-md">
            {renderContent()}
          </div>
        </div>
      </div>
    
    </div>
  );
}

export default VerifyEmail;
