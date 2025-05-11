import React, { useState } from 'react';
import axios from 'axios';
import Header from '../components/Header';
import { config } from '../config';

const API_BASE_URL = config.API_BASE_URL;

function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/request-password-reset/`, {
        email: email
      });

      if (response.data.status === 'success') {
        setSuccess('Password reset instructions have been sent to your email.');
        setEmail('');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'An error occurred. Please try again.');
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
            {/* Left Column - Header Text */}
            <div className="md:w-1/2 text-left">
              <h1 className="text-5xl lg:text-6xl font-bold text-white">
                Forgot your{' '}
                <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">
                  password?
                </span>
              </h1>
              <p className="text-gray-400 mt-6 text-xl">
                Don't worry! Enter your email address and we'll send you instructions to reset your password.
              </p>
            </div>

            {/* Right Column - Form */}
            <div className="md:w-1/2 w-full">
              <div className="bg-zinc-900 p-8 rounded-xl shadow-xl">
                {error && (
                  <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
                    {error}
                  </div>
                )}
                
                {success && (
                  <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-4">
                    {success}
                  </div>
                )}

                <form onSubmit={handleSubmit}>
                  <div className="mb-6">
                    <label htmlFor="email" className="block text-gray-300 mb-2">
                      Email Address
                    </label>
                    <input
                      type="email"
                      id="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full px-4 py-2 rounded bg-zinc-800 text-white border border-zinc-700 focus:outline-none focus:border-purple-500"
                      required
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className={`w-full bg-gradient-to-r from-purple-700 to-blue-700 text-white py-2 rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 ${
                      loading ? 'opacity-50 cursor-not-allowed' : ''
                    }`}
                  >
                    {loading ? 'Sending...' : 'Reset Password'}
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ForgotPassword;
