import gsap from 'gsap';
import { useEffect, useRef } from 'react';
import shopGif from '../assets/shop.gif';

function Home() {
    // Create refs for elements we want to animate
    const titleRef = useRef(null);
    const subtitleRef = useRef(null);
    const gradientTextRef = useRef(null);
    const buttonRef = useRef(null);
    const gifRef = useRef(null);

    useEffect(() => {
        // Create a timeline for smooth sequence animation
        const tl = gsap.timeline({ defaults: { ease: "power3.out" } });

        // Title animation
        tl.fromTo(titleRef.current,
            {
                x: -100,
                opacity: 0
            },
            {
                x: 0,
                opacity: 1,
                duration: 1
            }
        );

        // Gradient text animation
        tl.fromTo(gradientTextRef.current,
            {
                y: 30,
                opacity: 0
            },
            {
                y: 0,
                opacity: 1,
                duration: 0.8
            },
            "-=0.5" // Start slightly before previous animation ends
        );

        // Subtitle animation
        tl.fromTo(subtitleRef.current,
            {
                y: 20,
                opacity: 0
            },
            {
                y: 0,
                opacity: 1,
                duration: 0.8
            },
            "-=0.3"
        );

        // Button animation
        tl.fromTo(buttonRef.current,
            {
                scale: 0,
                opacity: 0
            },
            {
                scale: 1,
                opacity: 1,
                duration: 0.5,
                ease: "back.out(1.7)"
            },
            "-=0.2"
        );

        // GIF animation
        tl.fromTo(gifRef.current,
            {
                x: 100,
                opacity: 0,
                scale: 0.8
            },
            {
                x: 0,
                opacity: 1,
                scale: 1,
                duration: 1,
                ease: "power2.out"
            },
            "-=1" // Start while other animations are still playing
        );

        // Optional: Add hover animation for the button
        gsap.to(buttonRef.current, {
            scale: 1.05,
            duration: 0.3,
            paused: true,
            ease: "power2.out"
        }).reverse();

        // Cleanup function
        return () => {
            tl.kill();
        };
    }, []);

    return (
        <div id="home" className="bg-black min-h-screen relative w-full flex flex-col md:flex-row items-center md:justify-between px-4 md:px-28 overflow-x-hidden">
            <div className="text-white space-y-6 w-full md:w-1/2 pt-32 md:pt-0 md:py-8 text-center md:text-left">
                <h1 className="text-4xl sm:text-5xl md:text-7xl font-bold">
                    <span ref={titleRef}>Retail Vista</span>
                    <br /> 
                    <span 
                        ref={gradientTextRef}
                        className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text"
                    >
                        One Stop Solution
                    </span>
                </h1>
                <br />
                <h2 
                    ref={subtitleRef}
                    className="text-xl sm:text-2xl text-gray-300"
                >
                    Prevent Loss, Drive Insights, Enhance Experience!                </h2>
                <br />
                <div className="flex justify-center md:justify-start mb-8 md:mb-0">
                    <button 
                        ref={buttonRef}
                        className="bg-gradient-to-r from-purple-700 to-blue-700 text-white text-base sm:text-lg px-6 sm:px-8 py-3 rounded-full shadow-lg"
                    >
                        Discover Retail Vista
                    </button>
                </div>
            </div>
            <div className="w-full md:w-1/2 flex justify-center mt-24 md:mt-0">
                <img 
                    ref={gifRef}
                    src={shopGif}
                    alt="Retail Animation" 
                    className="w-full max-w-md md:max-w-xl rounded-3xl"
                />
            </div>
        </div>
    );
}

export default Home;