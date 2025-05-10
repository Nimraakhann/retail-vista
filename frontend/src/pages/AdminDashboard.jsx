import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DashboardHeader from '../components/DashboardHeader';
import axios from 'axios';
import Pagination from '../components/Pagination';

const AdminDashboard = () => {
  const navigate = useNavigate();
  const [activeSection, setActiveSection] = useState('users');
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [editingUser, setEditingUser] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);

  useEffect(() => {
    const accessToken = localStorage.getItem('accessToken');
    if (!accessToken) {
      navigate('/admin/login');
      return;
    }

    // Fetch all users with their profiles
    const fetchUsers = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/get_all_users/', {
          headers: {
            'Authorization': `Bearer ${accessToken}`
          }
        });

        if (response.data.status === 'success') {
          setUsers(response.data.users);
        } else {
          setError('Failed to fetch users');
        }
      } catch (err) {
        console.error('Error fetching users:', err);
        setError(err.response?.data?.message || 'An error occurred while fetching users');
      } finally {
        setLoading(false);
      }
    };

    fetchUsers();
  }, [navigate]);

  const handleDeleteUser = async (userId) => {
    const accessToken = localStorage.getItem('accessToken');
    
    try {
      const response = await axios.delete(`http://localhost:8000/api/delete_user/${userId}/`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });

      if (response.data.status === 'success') {
        setUsers(users.filter(user => user.id !== userId));
        alert('User deleted successfully');
      } else {
        alert(response.data.message || 'Failed to delete user');
      }
    } catch (err) {
      console.error('Error deleting user:', err);
      alert(err.response?.data?.message || 'An error occurred while deleting the user');
    }
  };

  const handleEditUser = async (userId) => {
    const accessToken = localStorage.getItem('accessToken');
    
    if (!editingUser) {
      const userToEdit = users.find(user => user.id === userId);
      setEditingUser(userToEdit);
    } else {
      try {
        const userData = {
          first_name: editingUser.first_name,
          last_name: editingUser.last_name,
          email: editingUser.email,
          title: editingUser.title || '',
          company: editingUser.company || '',
          phone: editingUser.phone || '',
          email_verified: editingUser.email_verified
        };

        const response = await axios.post(
          'http://localhost:8000/api/admin/update_user/', 
          { user_id: userId, ...userData },
          {
            headers: {
              'Authorization': `Bearer ${accessToken}`,
              'Content-Type': 'application/json'
            }
          }
        );

        if (response.data.status === 'success') {
          setUsers(users.map(user => 
            user.id === userId ? { ...user, ...userData } : user
          ));
          setEditingUser(null);
          alert('User updated successfully');
        } else {
          alert(response.data.message || 'Failed to update user');
        }
      } catch (err) {
        console.error('Error updating user:', err);
        alert(err.response?.data?.message || 'An error occurred while updating the user');
      }
    }
  };

  const cancelEdit = () => {
    setEditingUser(null);
  };

  const filteredUsers = users.filter(user => {
    const nameMatch = `${user.first_name} ${user.last_name}`.toLowerCase().includes(searchQuery.toLowerCase());
    const statusMatch = filterStatus === 'all' ? true : 
      (filterStatus === 'active' ? user.email_verified : !user.email_verified);
    return nameMatch && statusMatch;
  });

  const sidebarMenuItems = [
    { 
      name: 'Users', 
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a8 8 0 00-8 8h16a8 8 0 00-8-8z" />
        </svg>
      ), 
      section: 'users' 
    }
  ];

  const getRoleBadgeClasses = (role) => {
    switch(role) {
      case 'Admin':
        return 'bg-red-100 text-red-800';
      case 'Manager':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-green-100 text-green-800';
    }
  };

  // Calculate pagination
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = filteredUsers.slice(indexOfFirstItem, indexOfLastItem);

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  const handleItemsPerPageChange = (newSize) => {
    setItemsPerPage(newSize);
    setCurrentPage(1);
  };

  return (
    <div className="flex flex-col h-screen">
      <DashboardHeader />
      <div className="flex flex-1 overflow-hidden">
        {/* Vertical Sidebar */}
        <div className="w-20 bg-zinc-900 flex flex-col items-center py-8 space-y-4">
          {sidebarMenuItems.map((item) => (
            <button
              key={item.name}
              onClick={() => setActiveSection(item.section)}
              className={`
                p-3 rounded-lg transition-all duration-300 
                ${activeSection === item.section 
                  ? 'bg-purple-700 text-white' 
                  : 'text-gray-400 hover:bg-zinc-800 hover:text-white'}
              `}
            >
              {item.icon}
            </button>
          ))}
        </div>

        {/* Main Content Area */}
        <div className="flex-1 p-10 overflow-auto">
          <div className="bg-white shadow-xl rounded-2xl h-full overflow-hidden">
            {activeSection === 'users' && (
              <div className="flex flex-col h-full">
                <div className="bg-gradient-to-r from-purple-700 to-blue-700 p-6">
                  <h1 className="text-3xl font-bold text-white">User Management</h1>
                </div>
                <div className="flex-1 overflow-auto p-6">
                  {/* Search and Filter Bar */}
                  <div className="mb-4 flex flex-wrap gap-4 items-center">
                    <div className="relative flex-1">
                      <input
                        type="text"
                        placeholder="Search by name..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full p-3 pl-10 rounded-lg border border-gray-300 focus:outline-none focus:border-purple-500"
                      />
                      <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                      </div>
                    </div>
                    
                    <div className="flex gap-2">
                      <button
                        onClick={() => setFilterStatus('all')}
                        className={`px-4 py-2 rounded-lg ${filterStatus === 'all' 
                          ? 'bg-purple-600 text-white' 
                          : 'bg-gray-200 text-gray-700'}`}
                      >
                        All
                      </button>
                      <button
                        onClick={() => setFilterStatus('active')}
                        className={`px-4 py-2 rounded-lg ${filterStatus === 'active' 
                          ? 'bg-purple-600 text-white' 
                          : 'bg-gray-200 text-gray-700'}`}
                      >
                        Active
                      </button>
                      <button
                        onClick={() => setFilterStatus('pending')}
                        className={`px-4 py-2 rounded-lg ${filterStatus === 'pending' 
                          ? 'bg-purple-600 text-white' 
                          : 'bg-gray-200 text-gray-700'}`}
                      >
                        Pending
                      </button>
                    </div>
                  </div>

                  {loading ? (
                    <div className="flex justify-center items-center h-full">
                      <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-purple-600"></div>
                    </div>
                  ) : error ? (
                    <div className="text-red-600 text-center">{error}</div>
                  ) : filteredUsers.length === 0 ? (
                    <div className="text-center text-gray-500">
                      {users.length === 0 ? "No users found" : "No users match your search"}
                    </div>
                  ) : (
                    <>
                      <table className="w-full text-sm text-left text-gray-500">
                        <thead className="text-xs text-gray-700 uppercase bg-gray-50 sticky top-0">
                          <tr>
                            <th className="px-6 py-3">Store ID</th>
                            <th className="px-6 py-3">Name</th>
                            <th className="px-6 py-3">Email</th>
                            <th className="px-6 py-3">Title</th>
                            <th className="px-6 py-3">Company</th>
                            <th className="px-6 py-3">Status</th>
                            <th className="px-6 py-3">Created</th>
                            <th className="px-6 py-3">Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {currentItems.map((user) => (
                            <tr key={user.id} className="bg-white border-b hover:bg-gray-50">
                              <td className="px-6 py-4 font-medium text-gray-900">
                                {user.store_id || 'N/A'}
                              </td>
                              <td className="px-6 py-4 font-medium text-gray-900">
                                {editingUser?.id === user.id ? (
                                  <div className="flex gap-2">
                                    <input
                                      type="text"
                                      value={editingUser.first_name}
                                      onChange={(e) => setEditingUser({...editingUser, first_name: e.target.value})}
                                      className="w-1/2 px-2 py-1 border rounded"
                                    />
                                    <input
                                      type="text"
                                      value={editingUser.last_name}
                                      onChange={(e) => setEditingUser({...editingUser, last_name: e.target.value})}
                                      className="w-1/2 px-2 py-1 border rounded"
                                    />
                                  </div>
                                ) : (
                                  `${user.first_name} ${user.last_name}`
                                )}
                              </td>
                              <td className="px-6 py-4">
                                {editingUser?.id === user.id ? (
                                  <input
                                    type="email"
                                    value={editingUser.email}
                                    onChange={(e) => setEditingUser({...editingUser, email: e.target.value})}
                                    className="w-full px-2 py-1 border rounded"
                                  />
                                ) : (
                                  user.email
                                )}
                              </td>
                              <td className="px-6 py-4">
                                {editingUser?.id === user.id ? (
                                  <input
                                    type="text"
                                    value={editingUser.title || ''}
                                    onChange={(e) => setEditingUser({...editingUser, title: e.target.value})}
                                    className="w-full px-2 py-1 border rounded"
                                  />
                                ) : (
                                  user.title || 'N/A'
                                )}
                              </td>
                              <td className="px-6 py-4">
                                {editingUser?.id === user.id ? (
                                  <input
                                    type="text"
                                    value={editingUser.company || ''}
                                    onChange={(e) => setEditingUser({...editingUser, company: e.target.value})}
                                    className="w-full px-2 py-1 border rounded"
                                  />
                                ) : (
                                  user.company || 'N/A'
                                )}
                              </td>
                              <td className="px-6 py-4">
                                <span className={`
                                  px-2 py-1 rounded-full text-xs font-semibold
                                  ${user.email_verified ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}
                                `}>
                                  {user.email_verified ? 'Verified' : 'Pending'}
                                </span>
                              </td>
                              <td className="px-6 py-4">
                                {new Date(user.created_at).toLocaleDateString()}
                              </td>
                              <td className="px-6 py-4">
                                <div className="flex space-x-2">
                                  {editingUser?.id === user.id ? (
                                    <>
                                      <button 
                                        onClick={() => handleEditUser(user.id)}
                                        className="text-green-600 hover:text-green-900"
                                      >
                                        Save
                                      </button>
                                      <button 
                                        onClick={cancelEdit}
                                        className="text-gray-600 hover:text-gray-900"
                                      >
                                        Cancel
                                      </button>
                                    </>
                                  ) : (
                                    <>
                                      <button 
                                        onClick={() => handleEditUser(user.id)}
                                        className="text-blue-600 hover:text-blue-900"
                                      >
                                        Edit
                                      </button>
                                      <button 
                                        onClick={() => handleDeleteUser(user.id)}
                                        className="text-red-600 hover:text-red-900"
                                      >
                                        Delete
                                      </button>
                                    </>
                                  )}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>

                      <div className="[&_select]:bg-purple-700 [&_select]:text-white [&_option]:bg-purple-700">
                        <Pagination
                          totalItems={filteredUsers.length}
                          itemsPerPage={itemsPerPage}
                          currentPage={currentPage}
                          onPageChange={handlePageChange}
                          onItemsPerPageChange={handleItemsPerPageChange}
                        />
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
