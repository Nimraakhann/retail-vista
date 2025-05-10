import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DashboardHeader from '../components/DashboardHeader';
import { useAuth } from '../context/AuthContext';

function Profile() {
  const navigate = useNavigate();
  const { user, updateProfile, loading } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
    email: '',
    title: '',
    company: '',
    password: '',
    confirm_password: ''
  });

  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }

    setFormData({
      first_name: user.first_name || '',
      last_name: user.last_name || '',
      email: user.email || '',
      title: user.title || '',
      company: user.company || '',
      password: '',
      confirm_password: ''
    });
  }, [user, navigate]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    
    // Validation patterns
    const namePattern = /^[A-Za-z\s]*$/;  // Only letters and spaces
    const companyTitlePattern = /^[A-Za-z\s&'-]*$/;  // Letters, spaces, &, apostrophe, and hyphen
    
    // Validate based on field name
    if ((name === 'first_name' || name === 'last_name') && !namePattern.test(value)) {
      return; // Don't update if invalid
    }
    
    if ((name === 'company' || name === 'title') && !companyTitlePattern.test(value)) {
      return; // Don't update if invalid
    }
    
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage({ type: '', text: '' });

    // Client-side validation
    if (formData.password || formData.confirm_password) {
      if (formData.password !== formData.confirm_password) {
        setMessage({ type: 'error', text: 'Passwords do not match' });
        return;
      }
      if (formData.password.length < 8) {
        setMessage({ type: 'error', text: 'Password must be at least 8 characters long' });
        return;
      }
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      setMessage({ type: 'error', text: 'Please enter a valid email address' });
      return;
    }

    // Required fields validation
    if (!formData.first_name.trim() || !formData.last_name.trim() || !formData.email.trim()) {
      setMessage({ type: 'error', text: 'First name, last name, and email are required' });
      return;
    }

    // Remove confirm_password and empty password from submission
    const submitData = { ...formData };
    delete submitData.confirm_password;
    if (!submitData.password) delete submitData.password;

    try {
      const result = await updateProfile(submitData);
      
      if (result.success) {
        setMessage({ type: 'success', text: 'Profile updated successfully' });
        setIsEditing(false);
        // Reset password fields
        setFormData(prev => ({
          ...prev,
          password: '',
          confirm_password: ''
        }));
      } else {
        setMessage({ type: 'error', text: result.message });
      }
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error.response?.data?.message || 'An error occurred while updating your profile'
      });
    }
  };

  if (!user) {
    return null;
  }

  return (
    <div className="flex flex-col min-h-screen bg-black">
      <DashboardHeader />
      
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-4xl">
          {/* Profile Header */}
       
                <button
                  onClick={() => navigate(-1)}
                  className="text-white hover:text-purple-500 transition-colors"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                  </svg>
                </button>
             
          <div className="text-center mb-4">
          <h1 className="text-3xl font-bold text-white">
              Your <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">Profile</span>
            </h1>
            <p className="text-gray-400 text-sm">
              {isEditing ? 'Edit your profile information below' : 'Manage your personal information and preferences'}
            </p>
            {message.text && (
              <div className={`mt-2 text-sm ${message.type === 'success' ? 'text-green-400' : 'text-red-400'}`}>
                {message.text}
              </div>
            )}
          </div>

          {/* Profile Card */}
          <div className="bg-gradient-to-br from-zinc-900 to-zinc-800 rounded-2xl p-4 md:p-6 shadow-xl max-w-3xl mx-auto">
            <form onSubmit={handleSubmit}>
              <div className="grid grid-cols-12 gap-4 md:gap-6">
                {/* Details Section */}
                <div className="col-span-12 grid grid-cols-1 md:grid-cols-2 gap-4">
                  {isEditing ? (
                    <>
                      {/* Personal Information */}
                      <div>
                        <h3 className="text-sm font-semibold text-purple-400 mb-2">Personal Information</h3>
                        <div className="space-y-3">
                          <div>
                            <label className="block text-sm text-gray-400 mb-1">First Name</label>
                            <input
                              type="text"
                              name="first_name"
                              value={formData.first_name}
                              onChange={handleChange}
                              className="w-full bg-zinc-800 text-white text-sm rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                              required
                            />
                          </div>
                          <div>
                            <label className="block text-sm text-gray-400 mb-1">Last Name</label>
                            <input
                              type="text"
                              name="last_name"
                              value={formData.last_name}
                              onChange={handleChange}
                              className="w-full bg-zinc-800 text-white text-sm rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                              required
                            />
                          </div>
                        </div>
                      </div>

                      {/* Contact & Professional Details */}
                      <div className="space-y-4">
                        <div>
                          <h3 className="text-sm font-semibold text-purple-400 mb-2">Contact Details</h3>
                          <div>
                            <label className="block text-sm text-gray-400 mb-1">Email</label>
                            <input
                              type="email"
                              name="email"
                              value={formData.email}
                              onChange={handleChange}
                              className="w-full bg-zinc-800 text-white text-sm rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                              required
                            />
                          </div>
                        </div>
                        <div>
                          <h3 className="text-sm font-semibold text-purple-400 mb-2">Professional Details</h3>
                          <div className="space-y-3">
                            <div>
                              <label className="block text-sm text-gray-400 mb-1">Title</label>
                              <input
                                type="text"
                                name="title"
                                value={formData.title}
                                onChange={handleChange}
                                className="w-full bg-zinc-800 text-white text-sm rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                              />
                            </div>
                            <div>
                              <label className="block text-sm text-gray-400 mb-1">Company</label>
                              <input
                                type="text"
                                name="company"
                                value={formData.company}
                                onChange={handleChange}
                                className="w-full bg-zinc-800 text-white text-sm rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                              />
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Password Section */}
                      <div className="col-span-full">
                        <h3 className="text-sm font-semibold text-purple-400 mb-2">Change Password (Optional)</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div>
                            <label className="block text-sm text-gray-400 mb-1">New Password</label>
                            <input
                              type="password"
                              name="password"
                              value={formData.password}
                              onChange={handleChange}
                              className="w-full bg-zinc-800 text-white text-sm rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                              placeholder="Leave blank to keep current"
                            />
                          </div>
                          <div>
                            <label className="block text-sm text-gray-400 mb-1">Confirm Password</label>
                            <input
                              type="password"
                              name="confirm_password"
                              value={formData.confirm_password}
                              onChange={handleChange}
                              className="w-full bg-zinc-800 text-white text-sm rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                              placeholder="Leave blank to keep current"
                            />
                          </div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      {/* View Mode - Display Information in Table */}
                      <div className="col-span-12">
                        <table className="w-full">
                          <tbody className="divide-y divide-zinc-700">
                            <tr>
                              <td className="py-3 text-sm font-semibold text-purple-400 w-1/4">Name</td>
                              <td className="py-3 text-sm text-gray-400">{user.first_name} {user.last_name}</td>
                            </tr>
                            <tr>
                              <td className="py-3 text-sm font-semibold text-purple-400 w-1/4">Email</td>
                              <td className="py-3 text-sm text-gray-400">{user.email}</td>
                            </tr>
                            <tr>
                              <td className="py-3 text-sm font-semibold text-purple-400">Title</td>
                              <td className="py-3 text-sm text-gray-400">{user.title || 'Not specified'}</td>
                            </tr>
                            <tr>
                              <td className="py-3 text-sm font-semibold text-purple-400">Company</td>
                              <td className="py-3 text-sm text-gray-400">{user.company || 'Not specified'}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="col-span-12 flex justify-end space-x-4 mt-4">
                  {isEditing ? (
                    <>
                      <button
                        type="button"
                        onClick={() => {
                          setIsEditing(false);
                          setMessage({ type: '', text: '' });
                          setFormData({
                            first_name: user.first_name || '',
                            last_name: user.last_name || '',
                            email: user.email || '',
                            title: user.title || '',
                            company: user.company || '',
                            password: '',
                            confirm_password: ''
                          });
                        }}
                        className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-white bg-zinc-800 rounded-lg transition-colors"
                        disabled={loading}
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg hover:from-purple-700 hover:to-blue-700 transition-colors"
                        disabled={loading}
                      >
                        {loading ? 'Saving...' : 'Save Changes'}
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      onClick={() => {
                        setIsEditing(true);
                        setMessage({ type: '', text: '' });
                      }}
                      className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg hover:from-purple-700 hover:to-blue-700 transition-colors"
                    >
                      Edit Profile
                    </button>
                  )}
                </div>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Profile;
