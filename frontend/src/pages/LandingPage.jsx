import About from '../components/About'
import Contact from '../components/Contact'
import Faqs from '../components/Faqs'
import Footer from '../components/Footer'
import Header from '../components/Header'
import Home from '../components/Home'
import Pose from '../components/Pose'
import Solutions from '../components/Solutions'

export default function LandingPage() {
  return (
    <div>
      <Header />
      <Home />
      <Solutions />
      <Pose />
      <About />
      <Faqs />
      <Contact />
      <Footer />
    </div>
  )
}