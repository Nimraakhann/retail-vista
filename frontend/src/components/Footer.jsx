import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const Footer = () => {
  const location = useLocation();
  const navigate = useNavigate();
  
  const handleNavigation = (sectionId) => {
    // If we're on login , navigate to home page with section hash
    if (location.pathname === '/login' || location.pathname === '/signup') {
      navigate(`/#${sectionId}`);
    } else {
      // Normal scroll behavior for main page
      const element = document.getElementById(sectionId);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <footer className="bg-white shadow-lg px-4 md:px-28 py-8">
      <div className="container mx-auto">
        {/* Top Section with Logo and Links */}
        <div className="flex flex-col md:flex-row justify-between gap-8 mb-8">
          {/* Logo and Description - Left Side */}
          <div className="md:w-1/3">
            <h2 className="text-xl font-bold mb-4">Retail Vista</h2>
            <p className="text-gray-600 mb-4">
              Enhancing Store Security and Customer Engagement with Smart Solutions.
            </p>
          </div>

          {/* Links Container - Right Side */}
          <div className="flex flex-col md:flex-row gap-8 md:w-2/3 md:justify-end">
            {/* Quick Links */}
            <div className="md:w-48">
              <h3 className="font-semibold mb-4">Quick Links</h3>
              <ul className="space-y-2">
                <li><button onClick={() => handleNavigation('home')} className="text-gray-600 hover:text-purple-700">Home</button></li>
                <li><button onClick={() => handleNavigation('solutions')} className="text-gray-600 hover:text-purple-700">Solutions</button></li>
                <li><button onClick={() => handleNavigation('about')} className="text-gray-600 hover:text-purple-700">About Us</button></li>
                <li><button onClick={() => handleNavigation('faqs')} className="text-gray-600 hover:text-purple-700">FAQ</button></li>
              </ul>
            </div>

            {/* Contact Info */}
            <div className="md:w-64">
              <h3 className="font-semibold mb-4">Contact</h3>
              <ul className="space-y-2">
                <li className="text-gray-600">Email: anishhamayoon@gmail.com</li>
                <li className="text-gray-600">Phone: +92 333 5069449</li>
                <li className="text-gray-600">Address: 123 Tech Street, Islamabad, PK</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Bottom Section with Copyright and Social Links */}
        <div className="border-t border-gray-200 pt-8">
          <div className="flex flex-col items-center space-y-4">
            <div className="flex space-x-6">
              <a href="#" className="text-gray-600 hover:text-purple-700">
                <i className="fab fa-facebook"></i>
              </a>
              <a href="#" className="text-gray-600 hover:text-purple-700">
                <i className="fab fa-twitter"></i>
              </a>
              <a href="#" className="text-gray-600 hover:text-purple-700">
                <i className="fab fa-linkedin"></i>
              </a>
              <a href="#" className="text-gray-600 hover:text-purple-700">
                <i className="fab fa-instagram"></i>
              </a>
            </div>
            <p className="text-gray-600 text-sm text-center">
              © 2024 Retail Vista. All rights reserved.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
