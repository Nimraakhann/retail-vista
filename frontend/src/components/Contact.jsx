import React, { useState } from 'react'
import axios from 'axios';

const Contact = () => {
    const [formData, setFormData] = useState({
        firstName: '',
        lastName: '',
        email: '',
        phone: '',
        message: ''
    });
    const [status, setStatus] = useState({ type: '', message: '' });
    const [loading, setLoading] = useState(false);

    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    const handleChange = (e) => {
        const { name, value } = e.target;
        // Validation patterns
    const namePattern = /^[A-Za-z\s]*$/;  // Only letters and spaces
    const phonePattern = /^[0-9]*$/;  // Only numbers
    
    // Validate based on field name
        if ((name === 'firstName' || name === 'lastName') && !namePattern.test(value)) {
            return; // Don't update if invalid
        }
        if (name === 'phone' && (!phonePattern.test(value) || value.length > 11)) {
            return; // Don't update if invalid or greater than 11 numbers
        }

        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setStatus({ type: '', message: '' });

        try {
            const response = await axios.post(`${API_BASE_URL}/api/contact/`, formData, {
                timeout: 10000  
            });
            
            setStatus({ type: 'success', message: response.data.message });
            setFormData({
                firstName: '',
                lastName: '',
                email: '',
                phone: '',
                message: ''
            });
        } catch (error) {
            if (error.code === 'ECONNABORTED') {
                setStatus({
                    type: 'error',
                    message: 'Request timed out. Please try again.'
                });
            } else {
                setStatus({
                    type: 'error',
                    message: error.response?.data?.message || 'Something went wrong. Please try again.'
                });
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <hr className="h-1 bg-gradient-to-r from-purple-700 to-blue-700 border-0" />
            <div id="contact" className="bg-black w-full text-white py-16 md:py-24">
                <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12 items-center p-4 md:p-8">
                    {/* Left Column */}
                    <div className="space-y-6 md:space-y-8 md:-mt-60 md:-ml-16">
                        <div>
                            <h3 className="text-lg md:text-1xl font-bold mb-2 md:mb-4 bg-gradient-to-r from-purple-700 to-blue-700 inline-block text-transparent bg-clip-text">Get in Touch</h3>
                            <h1 className="text-4xl md:text-6xl font-bold mb-4">Discover the Power of <span className="bg-gradient-to-r from-purple-700 to-blue-700 inline-block text-transparent bg-clip-text">Retail Vista</span></h1>
                            <p className="text-gray-300">
                            Have questions or need more information? Reach out to us, and we'll be happy to assist you with your queries or demo requests.</p>
                        </div>
                    </div>

                    {/* Right Column - Contact Form */}
                    <div className="bg-zinc-900 p-6 md:p-8 rounded-lg">
                        <h2 className="text-xl md:text-2xl font-bold mb-6 text-center">Contact Us</h2>
                        {status.message && (
                            <div className={`p-4 rounded mb-4 ${status.type === 'success' ? 'bg-green-900 text-green-200' : 'bg-red-900 text-red-200'}`}>
                                {status.message}
                            </div>
                        )}
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium mb-2">First Name *</label>
                                    <input 
                                        type="text"
                                        name="firstName"
                                        value={formData.firstName}
                                        onChange={handleChange}
                                        required
                                        className="w-full p-2 rounded bg-zinc-800 border border-zinc-700 focus:outline-none focus:border-purple-500"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium mb-2">Last Name *</label>
                                    <input 
                                        type="text"
                                        name="lastName"
                                        value={formData.lastName}
                                        onChange={handleChange}
                                        required
                                        className="w-full p-2 rounded bg-zinc-800 border border-zinc-700 focus:outline-none focus:border-purple-500"
                                    />
                                </div>
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-2">Email *</label>
                                <input 
                                    type="email"
                                    name="email"
                                    value={formData.email}
                                    onChange={handleChange}
                                    required
                                    className="w-full p-2 rounded bg-zinc-800 border border-zinc-700 focus:outline-none focus:border-purple-500"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-2">Phone Number</label>
                                <input 
                                    type="tel"
                                    name="phone"
                                    value={formData.phone}
                                    onChange={handleChange}
                                    className="w-full p-2 rounded bg-zinc-800 border border-zinc-700 focus:outline-none focus:border-purple-500"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-2">Message *</label>
                                <textarea 
                                    rows="4"
                                    name="message"
                                    value={formData.message}
                                    onChange={handleChange}
                                    required
                                    className="w-full p-2 rounded bg-zinc-800 border border-zinc-700 focus:outline-none focus:border-purple-500"
                                ></textarea>
                            </div>
                            <button 
                                type="submit"
                                disabled={loading}
                                className={`w-full py-3 px-4 rounded font-semibold text-white bg-gradient-to-r from-purple-700 to-blue-700 hover:from-purple-600 hover:to-blue-600 transition-all duration-200 ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                                {loading ? 'Sending...' : 'Send Message'}
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </>
    )
}

export default Contact
