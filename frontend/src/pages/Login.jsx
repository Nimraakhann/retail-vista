import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const success = await login(email, password, rememberMe);
      if (success) {
        navigate('/dashboard', { replace: true });
      }
    } catch (err) {
      console.error('Login error:', err);
      setError(err.response?.data?.message || 'An error occurred during login');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative w-full overflow-x-hidden">
      <Header loginPage={true} />
      
      <div className="bg-black min-h-screen w-full">
        <div className="max-w-7xl mx-auto px-4 py-16 md:px-28">
          <div className="flex flex-col md:flex-row justify-between items-center gap-16">
            {/* Left Column - Header Text */}
            <div className="md:w-1/2 text-left">
              <h1 className="text-5xl lg:text-6xl font-bold text-white">
                Welcome back to{' '}
                <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">
                  Retail Vista
                </span>
              </h1>
              <p className="text-gray-400 mt-6 text-xl">
                Log in to access your dashboard and manage your retail security solutions.
              </p>
            </div>

            {/* Right Column - Login Form */}
            <div className="md:w-1/2 w-full max-w-md">
              <div className="bg-zinc-900 p-8 rounded-lg">
                <h2 className="text-2xl font-bold mb-6 text-white text-center">Login</h2>
                <form onSubmit={handleSubmit} className="space-y-6">
                  {error && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                      {error}
                    </div>
                  )}
                  <div className="space-y-4">
                    <div>
                      <label htmlFor="email" className="block text-sm text-gray-300">Email</label>
                      <input
                        type="email"
                        id="email"
                        name="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="w-full p-2 mt-1 bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-purple-500"
                        required
                      />
                    </div>
                    <div>
                      <label htmlFor="password" className="block text-sm text-gray-300">Password</label>
                      <div className="relative">
                        <input
                          type={showPassword ? 'password' : 'text'}
                          id="password"
                          name="password"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          className="w-full p-2 mt-1 bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-purple-500 pr-10"
                          required
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute inset-y-0 right-0 pr-3 flex items-center"
                        >
                          {showPassword ? (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                            </svg>
                          ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                            </svg>
                          )}
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <input
                        id="remember-me"
                        name="remember-me"
                        type="checkbox"
                        checked={rememberMe}
                        onChange={(e) => setRememberMe(e.target.checked)}
                        className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                      />
                      <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-300">
                        Remember me
                      </label>
                    </div>
                    <div className="flex items-center">
                      <Link to="/forgot-password" className="text-purple-500 hover:text-purple-400">
                        Forgot your password?
                      </Link>
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-gradient-to-r from-purple-700 to-blue-700 hover:from-purple-600 hover:to-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 disabled:opacity-50 transition-all duration-200"
                  >
                    {loading ? 'Signing in...' : 'Sign in'}
                  </button>

                  <p className="mt-6 text-center text-gray-400">
                    Don't have an account?{' '}
                    <Link to="/signup" className="text-purple-500 hover:text-purple-400">
                      Signup
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
};

export default Login;
