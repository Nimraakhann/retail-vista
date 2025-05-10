import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate, useLocation } from 'react-router-dom';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const checkAuth = async () => {
      const userData = localStorage.getItem('userData');
      if (userData) {
        setUser(JSON.parse(userData));
        if (location.pathname === '/login' || location.pathname === '/') {
          navigate('/dashboard');
        }
        return;
      }

      const rememberedAuth = localStorage.getItem('rememberedAuth');
      if (rememberedAuth && !user) {
        const { email, password } = JSON.parse(rememberedAuth);
        const success = await login(email, password, true);
        if (success) {
          
            navigate('/dashboard');
         
        }
      }
    };

    checkAuth();
  }, [navigate, location.pathname]);

  const login = async (email, password, rememberMe = false) => {
    try {
      const response = await axios.post('http://localhost:8000/api/login/', {
        email,
        password,
      });

      if (response.data.status === 'success') {
        localStorage.setItem('accessToken', response.data.tokens.access);
        localStorage.setItem('refreshToken', response.data.tokens.refresh);
        
        const userData = {
          id: response.data.user.id,
          email: response.data.user.email,
          username: response.data.user.username,
          first_name: response.data.user.first_name,
          last_name: response.data.user.last_name,
          title: response.data.user.title,
          company: response.data.user.company
        };
        localStorage.setItem('userData', JSON.stringify(userData));
        
        if (rememberMe) {
          localStorage.setItem('rememberedAuth', JSON.stringify({ email, password }));
        } else {
          localStorage.removeItem('rememberedAuth');
        }

        setUser(userData);
        return true;
      }
      throw new Error(response.data.message || 'Login failed');
    } catch (error) {
      if (error.response?.data?.message) {
        throw error;
      }
      throw new Error('An error occurred during login');
    }
  };

  const updateProfile = async (profileData) => {
    setLoading(true);
    setError(null);
    
    try {
      const accessToken = localStorage.getItem('accessToken');
      const response = await axios.post(
        'http://localhost:8000/api/update-profile/',
        profileData,
        {
          headers: {
            'Authorization': `Bearer ${accessToken}`
          }
        }
      );

      if (response.data.status === 'success') {
        const updatedUserData = response.data.user;
        localStorage.setItem('userData', JSON.stringify(updatedUserData));
        setUser(updatedUserData);
        setLoading(false);
        return { success: true, message: 'Profile updated successfully' };
      }
    } catch (error) {
      setError(error.response?.data?.message || 'An error occurred while updating profile');
      setLoading(false);
      return { 
        success: false, 
        message: error.response?.data?.message || 'An error occurred while updating profile'
      };
    }
  };

  const logout = () => {
    localStorage.removeItem('userData');
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
    localStorage.removeItem('rememberedAuth');
    setUser(null);
  };

  const value = {
    user,
    login,
    logout,
    updateProfile,
    loading,
    error
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;
