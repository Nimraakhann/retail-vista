import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';

function Header({ simplified = false }) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const location = useLocation();

  const isProfilePage = location.pathname === '/profile';
  const isHomePage = location.pathname === '/';
  const isLoginPage = location.pathname === '/login';
  const isSignupPage = location.pathname === '/signup';

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <>
      <nav className={`
        bg-white transition-all duration-300 px-4 md:px-8 lg:px-28 py-4 w-full fixed top-0 z-50
        ${isScrolled ? 'shadow-lg' : ''}
      `}>
        <div className="container mx-auto flex flex-wrap items-center justify-between">
          {/* Logo/Brand */}
          <Link to="/" className="text-xl font-bold">
            Retail Vista
          </Link>

          {/* Mobile Menu Button */}
          <button 
            className="md:hidden text-gray-600"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {isMenuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>

          {/* Mobile Menu */}
          <div className={`${isMenuOpen ? 'flex' : 'hidden'} md:hidden w-full flex-col absolute top-16 left-0 right-0 bg-white shadow-lg py-4 space-y-4 z-50`}>
            {!simplified && isHomePage && (
              <>
                <button onClick={() => scrollToSection('home')} className="text-black hover:text-custom-purple px-4 text-center w-full">
                  HOME
                </button>
                <button onClick={() => scrollToSection('solutions')} className="text-black hover:text-custom-purple px-4 text-center w-full">
                  SOLUTIONS
                </button>
                <button onClick={() => scrollToSection('about')} className="text-black hover:text-custom-purple px-4 text-center w-full">
                  ABOUT US
                </button>
                <button onClick={() => scrollToSection('faqs')} className="text-black hover:text-custom-purple px-4 text-center w-full">
                  FAQs
                </button>
                <button onClick={() => scrollToSection('contact')} className="text-black hover:text-custom-purple px-4 text-center w-full">
                  CONTACT US
                </button>
              
              </>
            )}
            <div className="flex flex-col items-center space-y-2 px-4">
              {isHomePage && (
                <Link to="/signup" className="w-full text-center bg-gradient-to-r from-purple-700 to-blue-700 text-white px-6 py-2 rounded-full shadow-lg hover:scale-105 transform transition duration-200">
                  Signup
                </Link>
              )}
              {(isProfilePage || isSignupPage || isLoginPage) && (
                <Link to="/" className="w-full text-center bg-gradient-to-r from-purple-700 to-blue-700 text-white px-6 py-2 rounded-full shadow-lg hover:scale-105 transform transition duration-200">
                  {isLoginPage ? 'Back to Home' : 'Home'}
                </Link>
              )}
              {(isHomePage || isSignupPage) && (
                <Link to="/login" className="w-full text-center border-2 border-purple-700 text-purple-700 px-6 py-2 rounded-full shadow-lg hover:bg-purple-700 hover:text-white transform transition duration-200">
                  Login
                </Link>
              )}
            </div>
          </div>

          {/* Desktop Menu and Buttons Container */}
          <div className="hidden md:flex md:flex-1 lg:flex-row items-center justify-between">
            {/* Desktop Menu */}
            <div className="flex items-center justify-center flex-1 space-x-8">
              {!simplified && isHomePage && (
                <>
                  <button onClick={() => scrollToSection('home')} className="text-black hover:text-custom-purple text-center">
                    HOME
                  </button>
                  <button onClick={() => scrollToSection('solutions')} className="text-black hover:text-custom-purple text-center">
                    SOLUTIONS
                  </button>
                  <button onClick={() => scrollToSection('about')} className="text-black hover:text-custom-purple text-center">
                    ABOUT US
                  </button>
                  <button onClick={() => scrollToSection('faqs')} className="text-black hover:text-custom-purple text-center">
                    FAQs
                  </button>
                  <button onClick={() => scrollToSection('contact')} className="text-black hover:text-custom-purple text-center">
                    CONTACT US
                  </button>
                 
                </>
              )}
            </div>

            {/* Desktop Auth Buttons */}
            <div className="flex items-center space-x-4 lg:ml-auto">
              {isHomePage && (
                <Link to="/signup" className="bg-gradient-to-r from-purple-700 to-blue-700 text-white px-6 py-2 rounded-full shadow-lg hover:scale-105 transform transition duration-200">
                  Signup
                </Link>
              )}
              {(isProfilePage || isSignupPage || isLoginPage) && (
                <Link to="/" className="bg-gradient-to-r from-purple-700 to-blue-700 text-white px-6 py-2 rounded-full shadow-lg hover:scale-105 transform transition duration-200">
                  {isLoginPage ? 'Back to Home' : 'Home'}
                </Link>
              )}
              {(isHomePage || isSignupPage) && (
                <Link to="/login" className="border-2 border-purple-700 text-purple-700 px-6 py-2 rounded-full shadow-lg hover:bg-purple-700 hover:text-white transform transition duration-200">
                  Login
                </Link>
              )}
            </div>
          </div>
        </div>
      </nav>
      <div className="h-20"></div> {/* Spacer for fixed header */}
    </>
  );
}

export default Header;