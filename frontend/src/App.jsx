import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Signup from './pages/Signup';
import VerifyEmail from './pages/VerifyEmail';
import Profile from './pages/Profile';
import { AuthProvider } from './context/AuthContext';
import ForgotPassword from './pages/ForgotPassword';
import PrivateRoute from './components/PrivateRoute';
import AdminLogin from './pages/AdminLogin';
import AdminDashboard from './pages/AdminDashboard';
import ResetPassword from './pages/ResetPassword';
import ManagePromotions from './pages/ManagePromotions';
import DisplayPromotions from './pages/DisplayPromotions';
import PeopleCounter from './pages/peoplecounter'; 
import ShopliftingAlert from './components/ShopliftingAlert';
import RecentActivity from './pages/RecentActivity';

import UploadMap from './pages/UploadMap';
import ShowMap from './pages/ShowMap';
import ShopliftingDetection from './pages/ShopliftingDetection';
import EditMap from './pages/EditMap';
import AgeGenderDetection from './pages/AgeGenderDetection';

function App() {
  return (
    <Router>
      <AuthProvider>
        <div className="App">
          <ShopliftingAlert />
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
            <Route path="/reset-password" element={<ResetPassword />} />
            <Route path="/signup" element={<Signup />} />
            <Route path="/verify-email" element={<VerifyEmail />} />
            <Route path="/admin/login" element={<AdminLogin />} />
            <Route path="/admin/dashboard" element={<AdminDashboard />} />
            <Route 
              path="/dashboard" 
              element={
                <PrivateRoute>
                  <Dashboard />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/profile" 
              element={
                <PrivateRoute>
                  <Profile />
                </PrivateRoute>
              } 
            />
            
            <Route 
              path="/manage-promotions" 
              element={
                <PrivateRoute>
                  <ManagePromotions />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/display-promotions" 
              element={
                <PrivateRoute>
                  <DisplayPromotions />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/upload-map" 
              element={
                <PrivateRoute>
                  <UploadMap />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/show-map" 
              element={<ShowMap />}
            />
            <Route 
              path="/shoplifting-detection" 
              element={
                <PrivateRoute>
                  <ShopliftingDetection />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/shoplifting-detection/analysis" 
              element={
                <PrivateRoute>
                  <ShopliftingDetection />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/recent-activity" 
              element={
                <PrivateRoute>
                  <RecentActivity />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/edit-map" 
              element={
                <PrivateRoute>
                  <EditMap />
                </PrivateRoute>
              } 
            />
             
            <Route 
              path="/age-gender" 
              element={
                <PrivateRoute>
                  <AgeGenderDetection />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/age-gender/analysis" 
              element={
                <PrivateRoute>
                  <AgeGenderDetection />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/people-counter" 
              element={
                <PrivateRoute>
                  <PeopleCounter />
                </PrivateRoute>
              }
            />
            <Route 
              path="/people-counter/analysis" 
              element={
                <PrivateRoute>
                  <PeopleCounter />
                </PrivateRoute>
              }
            />
            
          </Routes>
        </div>
      </AuthProvider>
    </Router>
  );
}

export default App;
