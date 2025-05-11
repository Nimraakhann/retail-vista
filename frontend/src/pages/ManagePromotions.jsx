import React, { useState, useEffect } from 'react';
import DashboardHeader from '../components/DashboardHeader';
import Pagination from '../components/Pagination';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { config } from '../config';

const customFileInputStyle = `
  .custom-file-input::file-selector-button {
    background: linear-gradient(to right, #6b46c1, #3b82f6);
    border: 0;
    border-radius: 8px;
    color: white;
    cursor: pointer;
    padding: 8px 16px;
    margin-right: 16px;
    transition: all 0.2s;
  }
  .custom-file-input::file-selector-button:hover {
    background: linear-gradient(to right, #5b35b1, #2a71e6);
    transform: scale(1.05);
  }
`;

const API_BASE_URL = config.API_BASE_URL;

const ManagePromotions = () => {
  const [promotions, setPromotions] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [selectedPromotion, setSelectedPromotion] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    image: null,
    status: 'inactive'
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const navigate = useNavigate();

  // Fetch promotions
  useEffect(() => {
    fetchPromotions();
  }, []);

  const fetchPromotions = async () => {
    try {
      const accessToken = localStorage.getItem('accessToken');
      const response = await axios.get(`${API_BASE_URL}/api/promotions/`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });
      setPromotions(response.data.promotions);
    } catch (error) {
      console.error('Error fetching promotions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleImageChange = (e) => {
    setFormData({
      ...formData,
      image: e.target.files[0]
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formDataToSend = new FormData();
    formDataToSend.append('name', formData.name);
    if (formData.image) {
      formDataToSend.append('image', formData.image);
    }
    formDataToSend.append('status', formData.status);

    try {
      const accessToken = localStorage.getItem('accessToken');
      const config = {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'multipart/form-data'
        }
      };

      if (showEditModal) {
        await axios.put(
          `${API_BASE_URL}/api/promotions/${selectedPromotion.id}/`, 
          formDataToSend,
          config
        );
      } else {
        await axios.post(
          `${API_BASE_URL}/api/promotions/`,
          formDataToSend,
          config
        );
      }
      
      fetchPromotions();
      setShowAddModal(false);
      setShowEditModal(false);
      setFormData({ name: '', image: null, status: 'inactive' });
    } catch (error) {
      console.error('Error saving promotion:', error);
      alert(error.response?.data?.message || 'Error saving promotion');
    }
  };

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this promotion?')) {
      try {
        const accessToken = localStorage.getItem('accessToken');
        await axios.delete(`${API_BASE_URL}/api/promotions/${id}/`, {
          headers: {
            'Authorization': `Bearer ${accessToken}`
          }
        });
        fetchPromotions();
      } catch (error) {
        console.error('Error deleting promotion:', error);
        alert(error.response?.data?.message || 'Error deleting promotion');
      }
    }
  };

  const filteredPromotions = promotions.filter(promotion =>
    promotion.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Calculate pagination
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = filteredPromotions.slice(indexOfFirstItem, indexOfLastItem);

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  const handleItemsPerPageChange = (newSize) => {
    setItemsPerPage(newSize);
    setCurrentPage(1); // Reset to first page when changing items per page
  };

  return (
    <>
      <style>{customFileInputStyle}</style>
      <div className="min-h-screen bg-black">
        <DashboardHeader />
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="bg-zinc-900 rounded-lg shadow-xl p-6">
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => navigate(-1)}
                  className="text-white hover:text-purple-500 transition-colors"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                  </svg>
                </button>
                <h1 className="text-3xl font-bold text-white">
                 Manage <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">Promotions</span>
               </h1>
              </div>
              <button
                onClick={() => {
                  setFormData({ name: '', image: null, status: 'inactive' });
                  setShowAddModal(true);
                }}
                className="bg-gradient-to-r from-purple-600 to-blue-600  hover:from-purple-700 hover:to-blue-700 transition-colors text-white px-4 py-2 rounded-lg"
              >
                Add Promotion
              </button>
            </div>

            <div className="mb-4">
              <input
                type="text"
                placeholder="Search promotions..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full p-2 bg-zinc-800 border border-zinc-700 rounded text-white"
              />
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-gray-300">
                <thead className="text-xs uppercase bg-zinc-800">
                  <tr>
                    <th className="px-6 py-3">Name</th>
                    <th className="px-6 py-3">Preview</th>
                    <th className="px-6 py-3">Status</th>
                    <th className="px-6 py-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {currentItems.map((promotion) => (
                    <tr key={promotion.id} className="border-b border-zinc-800">
                      <td className="px-6 py-4">{promotion.name}</td>
                      <td className="px-6 py-4">
                        <img
                          src={promotion.image_url}
                          alt={promotion.name}
                          className="h-16 w-16 object-cover rounded"
                        />
                      </td>
                      <td className="px-6 py-4">
                        <span className={`px-2 py-1 rounded text-xs ${
                          promotion.status === 'active' 
                            ? 'bg-green-900 text-green-300' 
                            : 'bg-red-900 text-red-300'
                        }`}>
                          {promotion.status}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex space-x-2">
                          <button
                            onClick={() => {
                              setSelectedPromotion(promotion);
                              setFormData({
                                name: promotion.name,
                                status: promotion.status,
                                image: null
                              });
                              setShowEditModal(true);
                            }}
                            className="text-blue-400 hover:text-blue-300"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => handleDelete(promotion.id)}
                            className="text-red-400 hover:text-red-300"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <Pagination
              totalItems={filteredPromotions.length}
              itemsPerPage={itemsPerPage}
              currentPage={currentPage}
              onPageChange={handlePageChange}
              onItemsPerPageChange={handleItemsPerPageChange}
            />
          </div>
        </div>

        {/* Add/Edit Modal */}
        {(showAddModal || showEditModal) && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
            <div className="bg-zinc-900 rounded-lg p-6 max-w-md w-full">
              <h2 className="text-xl font-bold text-white mb-4">
                {showEditModal ? 'Edit Promotion' : 'Add New Promotion'}
              </h2>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Name</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                    className="w-full p-2 bg-zinc-800 border border-zinc-700 rounded text-white"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Image</label>
                  <input
                    type="file"
                    onChange={handleImageChange}
                    className="w-full p-2 bg-zinc-800 border border-zinc-700 rounded text-white custom-file-input"
                    accept="image/*"
                    required={!showEditModal}
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Status</label>
                  <select
                    value={formData.status}
                    onChange={(e) => setFormData({...formData, status: e.target.value})}
                    className="w-full p-2 bg-zinc-800 border border-zinc-700 rounded text-white"
                  >
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                  </select>
                </div>
                <div className="flex justify-end space-x-3">
                  <button
                    type="button"
                    onClick={() => {
                      setShowAddModal(false);
                      setShowEditModal(false);
                    }}
                    className="px-4 py-2 text-gray-400 hover:text-white"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded"
                  >
                    {showEditModal ? 'Save Changes' : 'Add Promotion'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default ManagePromotions;
