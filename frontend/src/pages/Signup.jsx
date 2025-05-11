import React, { useState } from 'react';
import Footer from '../components/Footer';
import Header from '../components/Header';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { config } from '../config';

const API_BASE_URL = config.API_BASE_URL;

function Signup() {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    title: '',
    email: '',
    company: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const [passwordValidation, setPasswordValidation] = useState({
    minLength: false,
    hasUpperCase: false,
    hasNumber: false,
    hasSpecial: false
  });
  const [showPassword, setShowPassword] = useState(false);

  const validatePassword = (password) => {
    setPasswordValidation({
      minLength: password.length >= 8,
      hasUpperCase: /[A-Z]/.test(password),
      hasNumber: /[0-9]/.test(password),
      hasSpecial: /[!@#$%^&*]/.test(password)
    });
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    
    // Validation patterns
    const namePattern = /^[A-Za-z\s]*$/;  // Only letters and spaces
    const companyTitlePattern = /^[A-Za-z\s&'-]*$/;  // Letters, spaces, &, apostrophe, and hyphen
    
    // Validate based on field name
    if ((name === 'firstName' || name === 'lastName') && !namePattern.test(value)) {
      return; // Don't update if invalid
    }
    
    if ((name === 'company' || name === 'title') && !companyTitlePattern.test(value)) {
      return; // Don't update if invalid
    }
    
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));

    if (name === 'password') {
      validatePassword(value);
    }
  };

  const isPasswordValid = () => {
    return Object.values(passwordValidation).every(Boolean);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!isPasswordValid()) {
      setError('Please ensure your password meets all requirements');
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/signup/`, formData);

      if (response.data.status === 'success') {
        setSuccess('Please check your email to verify your account.');
        setFormData({
          firstName: '',
          lastName: '',
          title: '',
          email: '',
          company: '',
          password: ''
        });
        setPasswordValidation({
          minLength: false,
          hasUpperCase: false,
          hasNumber: false,
          hasSpecial: false
        });
      }
    } catch (err) {
      setError(err.response?.data?.message || 'An error occurred during signup');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative w-full overflow-x-hidden">
      <Header simplified={true} />
      
      <div className="bg-black min-h-screen w-full">
        <div className="max-w-7xl mx-auto px-4 py-16 md:px-28">
          <div className="flex flex-col md:flex-row justify-between items-center gap-16">
            <div className="md:w-1/2 text-left transform -translate-y-32 relative bottom-24 -mt-20">
              <h1 className="text-5xl lg:text-6xl font-bold text-white">
                Start catching shoplifters.{' '}
                <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">
                  Sign up now.
                </span>
              </h1>
              <p className="text-gray-400 mt-6 text-xl">
                Create your account to access our retail solutions.
              </p>
            </div>

            <div className="md:w-1/2 w-full max-w-md">
              <div className="bg-zinc-900 p-8 rounded-lg">
                <h2 className="text-2xl font-bold mb-6 text-white text-center">Sign Up</h2>
                <form onSubmit={handleSubmit} className="space-y-6">
                  {error && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                      {error}
                    </div>
                  )}
                  {success && (
                    <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
                      {success}
                    </div>
                  )}
                  <div className="space-y-4">
                    <div>
                      <label htmlFor="firstName" className="block text-sm text-gray-300">First name</label>
                      <input 
                        type="text"
                        id="firstName"
                        name="firstName"
                        value={formData.firstName}
                        onChange={handleChange}
                        className="w-full p-2 mt-1 bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-purple-500"
                        required
                      />
                      <p className="text-xs text-gray-500 mt-1">Only letters are allowed</p>
                    </div>
                    <div>
                      <label htmlFor="lastName" className="block text-sm text-gray-300">Last name</label>
                      <input 
                        type="text"
                        id="lastName"
                        name="lastName"
                        value={formData.lastName}
                        onChange={handleChange}
                        className="w-full p-2 mt-1 bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-purple-500"
                        required
                      />
                      <p className="text-xs text-gray-500 mt-1">Only letters are allowed</p>
                    </div>
                    <div>
                      <label htmlFor="email" className="block text-sm text-gray-300">Work Email</label>
                      <input 
                        type="email"
                        id="email"
                        name="email"
                        value={formData.email}
                        onChange={handleChange}
                        className="w-full p-2 mt-1 bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-purple-500"
                        required
                      />
                    </div>
                    <div>
                      <label htmlFor="password" className="block text-sm text-gray-300">Password</label>
                      <div className="relative">
                        <input 
                          type={showPassword ? "text" : "password"}
                          id="password"
                          name="password"
                          value={formData.password}
                          onChange={handleChange}
                          className="w-full p-2 mt-1 bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-purple-500 pr-10"
                          required
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400"
                        >
                          {showPassword ? (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                            </svg>
                          ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                            </svg>
                          )}
                        </button>
                      </div>
                      <div className="mt-2 space-y-1">
                        <p className={`text-sm ${passwordValidation.minLength ? 'text-green-600' : 'text-gray-500'}`}>
                          ✓ Minimum 8 characters
                        </p>
                        <p className={`text-sm ${passwordValidation.hasUpperCase ? 'text-green-600' : 'text-gray-500'}`}>
                          ✓ At least one uppercase letter
                        </p>
                        <p className={`text-sm ${passwordValidation.hasNumber ? 'text-green-600' : 'text-gray-500'}`}>
                          ✓ At least one number
                        </p>
                        <p className={`text-sm ${passwordValidation.hasSpecial ? 'text-green-600' : 'text-gray-500'}`}>
                          ✓ At least one special character ( ! @ # $ % ^ & * )
                        </p>
                      </div>
                    </div>
                    <div>
                      <label htmlFor="title" className="block text-sm text-gray-300">Title</label>
                      <input 
                        type="text"
                        id="title"
                        name="title"
                        value={formData.title}
                        onChange={handleChange}
                        className="w-full p-2 mt-1 bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-purple-500"
                        placeholder="e.g., Store Manager, Owner"
                        required
                      />
                      <p className="text-xs text-gray-500 mt-1">Only letters, spaces, hyphens, and apostrophes allowed</p>
                    </div>
                    <div>
                      <label htmlFor="company" className="block text-sm text-gray-300">Company Name</label>
                      <input 
                        type="text"
                        id="company"
                        name="company"
                        value={formData.company}
                        onChange={handleChange}
                        className="w-full p-2 mt-1 bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-purple-500"
                        placeholder="Your company name"
                        required
                      />
                      <p className="text-xs text-gray-500 mt-1">Only letters, spaces, hyphens, and apostrophes allowed</p>
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-gradient-to-r from-purple-700 to-blue-700 hover:from-purple-600 hover:to-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 disabled:opacity-50 transition-all duration-200"
                  >
                    {loading ? 'Creating account...' : 'Create account'}
                  </button>

                  <p className="mt-6 text-center text-gray-400">
                    Already have an account?{' '}
                    <Link to="/login" className="text-purple-500 hover:text-purple-400">
                      Login
                    </Link>
                  </p>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
}

export default Signup;
