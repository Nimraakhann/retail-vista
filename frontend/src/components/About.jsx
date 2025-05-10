import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import aboutImg from '../assets/about.png';

gsap.registerPlugin(ScrollTrigger);

function About() {
  // Create refs for elements we want to animate
  const imageContainerRef = useRef(null);
  const imageBackgroundRef = useRef(null);
  const imageRef = useRef(null);
  const titleRef = useRef(null);
  const lineRef = useRef(null);
  const textRef = useRef(null);

  useEffect(() => {
    // Create timeline for content animations
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: imageContainerRef.current,
        start: "top 80%",
        toggleActions: "play none none reverse"
      }
    });

    // Image background animation
    tl.fromTo(imageBackgroundRef.current,
      {
        opacity: 0,
        x: -50,
      },
      {
        opacity: 1,
        x: 0,
        duration: 1,
        ease: "power3.out"
      }
    );

    // Image animation
    tl.fromTo(imageRef.current,
      {
        opacity: 0,
        x: -30,
        scale: 0.9,
      },
      {
        opacity: 1,
        x: 0,
        scale: 1,
        duration: 1.2,
        ease: "power3.out"
      },
      "-=0.8" // Start before previous animation ends
    );

    // Content animations
    tl.fromTo(titleRef.current,
      {
        opacity: 0,
        y: 30
      },
      {
        opacity: 1,
        y: 0,
        duration: 0.8,
        ease: "power3.out"
      },
      "-=0.5"
    );

    // Line animation
    tl.fromTo(lineRef.current,
      {
        scaleX: 0,
        transformOrigin: "left center"
      },
      {
        scaleX: 1,
        duration: 0.8,
        ease: "power3.inOut"
      },
      "-=0.5"
    );

    // Text animation
    tl.fromTo(textRef.current,
      {
        opacity: 0,
        y: 20
      },
      {
        opacity: 1,
        y: 0,
        duration: 0.8,
        ease: "power3.out"
      },
      "-=0.3"
    );

    // Cleanup
    return () => {
      ScrollTrigger.getAll().forEach(trigger => trigger.kill());
    };
  }, []);

  return (
    <div id="about" className="min-h-screen w-full flex items-center justify-center p-8">
      <hr className="h-1 bg-gradient-to-r from-purple-700 to-blue-700 border-0" />
      <div className="container mx-auto flex flex-col md:flex-row items-center gap-12">
        <div 
          ref={imageContainerRef}
          className="w-full md:w-1/2 relative -mt-12"
        >
          <div 
            ref={imageBackgroundRef}
            className="absolute top-8 left-1/2 right-[-20px] bottom-[-40px] bg-gray-100 rounded-lg" 
          />
          <img
            ref={imageRef}
            src={aboutImg}
            alt="About Us"
            className="rounded-lg shadow-lg w-full max-w-md mx-auto relative z-10"
          />
        </div>
        <div className="w-full md:w-1/2 space-y-6">
          <h1 
            ref={titleRef}
            className="text-4xl font-bold text-gray-800"
          >
            About Us
          </h1>
          <hr 
            ref={lineRef}
            className="h-1 w-32 bg-gradient-to-r from-purple-700 to-blue-700 border-0 my-4" 
          />
          <p 
            ref={textRef}
            className="text-lg text-gray-600"
          >
            At Retail Vista, we specialize in smart solutions that enhance store security and customer engagement, empowering retailers to create safer, more efficient, and customer-focused environments. From real-time shoplifting detection to advanced demographic analytics and foot traffic insights, we provide innovative tools to optimize operations and foster meaningful shopper experiences.
          </p>
        </div>
      </div>
    </div>
  );
}

export default About;