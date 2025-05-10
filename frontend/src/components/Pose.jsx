import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useEffect, useRef } from 'react';
import pose1 from '../assets/pose1.png';
import pose2 from '../assets/pose2.png';
import pose3 from '../assets/pose3.png';

gsap.registerPlugin(ScrollTrigger);

function Pose() {
    const titleRef = useRef(null);
    const cardsRef = useRef([]);

    useEffect(() => {
        // Title animation
        gsap.fromTo(titleRef.current,
            {
                opacity: 0,
                y: 50
            },
            {
                opacity: 1,
                y: 0,
                duration: 1,
                scrollTrigger: {
                    trigger: titleRef.current,
                    start: "top 80%",
                    toggleActions: "play none none reverse"
                }
            }
        );

        // Cards animation
        cardsRef.current.forEach((card, index) => {
            gsap.fromTo(card,
                {
                    opacity: 0,
                    x: -100,
                    scale: 0.8
                },
                {
                    opacity: 1,
                    x: 0,
                    scale: 1,
                    duration: 0.8,
                    delay: index * 0.3, // Stagger effect
                    ease: "power3.out",
                    scrollTrigger: {
                        trigger: card,
                        start: "top 85%",
                        toggleActions: "play none none reverse"
                    }
                }
            );
        });

        // Cleanup
        return () => {
            ScrollTrigger.getAll().forEach(trigger => trigger.kill());
        };
    }, []);

    // Function to add cards to refs array
    const addToRefs = (el) => {
        if (el && !cardsRef.current.includes(el)) {
            cardsRef.current.push(el);
        }
    };

    return (
        
        <div className="min-h-screen bg-black flex flex-col justify-center">
            
            <div className="container mx-auto px-4 md:px-36">
                <h1 
                    ref={titleRef}
                    className="text-4xl md:text-7xl font-bold mt-8 text-left md:text-center bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text"
                >
                    We care about you and{' '}
                    <span className="md:hidden">your customer's privacy.</span>
                    <span className="hidden md:inline">
                        <br /> your customer's privacy.
                    </span>
                </h1>
                
                <div className="flex flex-col md:flex-row justify-center items-center mt-16 md:gap-8 gap-12">
                    <div 
                        ref={addToRefs}
                        className="flex flex-col items-center"
                    >
                        <img src={pose1} alt="Pose 1" className="h-[90px] max-w-full object-cover rounded-lg" />
                        <h2 className="text-2xl font-semibold mt-4 text-white">Security First</h2>
                        <p className="text-gray-400 text-center mt-2 max-w-sm">
                            Your data security is our top priority. We implement industry-leading encryption standards.
                        </p>
                    </div>

                    <div 
                        ref={addToRefs}
                        className="flex flex-col items-center"
                    >
                        <img src={pose2} alt="Pose 2" className="h-[90px] max-w-full object-cover rounded-lg" />
                        <h2 className="text-2xl font-semibold mt-4 text-white">Privacy Guaranteed</h2>
                        <p className="text-gray-400 text-center mt-2 max-w-sm">
                            We never share your personal information with third parties without your consent.
                        </p>
                    </div>

                    <div 
                        ref={addToRefs}
                        className="flex flex-col items-center"
                    >
                        <img src={pose3} alt="Pose 3" className="h-[90px] max-w-full object-cover rounded-lg" />
                        <h2 className="text-2xl font-semibold mt-4 text-white">Data Control</h2>
                        <p className="text-gray-400 text-center mt-2 max-w-sm">
                            You have full control over your data with easy management tools.
                        </p>
                    </div>
                    <br />
                </div>
            </div>
            
        </div>
    )
}

export default Pose