import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

function Solutions() {
    const cardsRef = useRef([]);
    const titleRef = useRef(null);

    useEffect(() => {
        // Animate the title
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

        // Animate each card
        cardsRef.current.forEach((card, index) => {
            gsap.fromTo(card,
                {
                    opacity: 0,
                    y: 50,
                    scale: 0.9
                },
                {
                    opacity: 1,
                    y: 0,
                    scale: 1,
                    duration: 0.8,
                    delay: index * 0.2, // Stagger effect
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
        <div id="solutions" className="min-h-screen bg-white">
            <hr className="h-1 bg-gradient-to-r from-purple-700 to-blue-700 border-0" />
            <h1 
                ref={titleRef}
                className="text-4xl md:text-7xl font-bold text-black mt-8 text-center"
            >
                State-of-the-art <span className="bg-gradient-to-r from-purple-700 to-blue-700 text-transparent bg-clip-text">AI</span> you can trust.
            </h1>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-12 px-4 md:px-28">
                {/* Card 1 */}
                <div ref={addToRefs} className="bg-gradient-to-r from-purple-700 to-blue-700 p-[1px] rounded-xl">
                    <div className="flex items-center p-6 bg-white rounded-xl">
                        <div className="flex-1">
                            <h3 className="text-xl font-bold text-gray-900">Real-Time Shoplifting Detection</h3>
                            <p className="text-gray-600 mt-4">Advanced computer vision system that monitors store activity and instantly alerts security personnel about potential shoplifting incidents. Trained on diverse retail environments for maximum accuracy.</p>
                        </div>
                        <div className="ml-6">
                            <svg className="w-12 h-12 text-purple-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
                            </svg>
                        </div>
                    </div>
                </div>

                {/* Card 2 */}
                <div ref={addToRefs} className="bg-gradient-to-r from-purple-700 to-blue-700 p-[1px] rounded-xl">
                    <div className="flex items-center p-6 bg-white rounded-xl">
                        <div className="flex-1">
                            <h3 className="text-xl font-bold text-gray-900">Notification System</h3>
                            <p className="text-gray-600 mt-4">Comprehensive alert system that delivers real-time notifications to staff devices with detailed incident information, including timestamps, location data, and high-resolution imagery.</p>
                        </div>
                        <div className="ml-6">
                            <svg className="w-12 h-12 text-purple-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"></path>
                            </svg>
                        </div>
                    </div>
                </div>

                {/* Card 3 */}
                <div ref={addToRefs} className="bg-gradient-to-r from-purple-700 to-blue-700 p-[1px] rounded-xl">
                    <div className="flex items-center p-6 bg-white rounded-xl">
                        <div className="flex-1">
                            <h3 className="text-xl font-bold text-gray-900">Demographic Data Analysis</h3>
                            <p className="text-gray-600 mt-4">Sophisticated analysis tools that process customer demographic data to provide valuable insights about shopping patterns, preferences, and peak activity times for better business decisions.</p>
                        </div>
                        <div className="ml-6">
                            <svg className="w-12 h-12 text-purple-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
                            </svg>
                        </div>
                    </div>
                </div>

                {/* Card 4 */}
                <div ref={addToRefs} className="bg-gradient-to-r from-purple-700 to-blue-700 p-[1px] rounded-xl">
                    <div className="flex items-center p-6 bg-white rounded-xl">
                        <div className="flex-1">
                            <h3 className="text-xl font-bold text-gray-900">Customer Traffic Analysis</h3>
                            <p className="text-gray-600 mt-4">Advanced tracking system utilizing AI algorithms to analyze customer movement patterns and generate detailed heat maps for optimizing store layout and product placement strategies.</p>
                        </div>
                        <div className="ml-6">
                            <svg className="w-12 h-12 text-purple-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                            </svg>
                        </div>
                    </div>
                </div>

                {/* Card 5 */}
                <div ref={addToRefs} className="bg-gradient-to-r from-purple-700 to-blue-700 p-[1px] rounded-xl">
                    <div className="flex items-center p-6 bg-white rounded-xl">
                        <div className="flex-1">
                            <h3 className="text-xl font-bold text-gray-900">Interactive Store Map</h3>
                            <p className="text-gray-600 mt-4">Dynamic digital mapping system that guides customers to products and departments, featuring real-time updates on store layout changes and temporary promotional displays.</p>
                        </div>
                        <div className="ml-6">
                            <svg className="w-12 h-12 text-purple-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"></path>
                            </svg>
                        </div>
                    </div>
                </div>

                {/* Card 6 */}
                <div ref={addToRefs} className="bg-gradient-to-r from-purple-700 to-blue-700 p-[1px] rounded-xl">
                    <div className="flex items-center p-6 bg-white rounded-xl">
                        <div className="flex-1">
                            <h3 className="text-xl font-bold text-gray-900">Store Promotions and Discounts</h3>
                            <p className="text-gray-600 mt-4">Intelligent promotion management system that automatically displays targeted offers and updates pricing information across all store displays, enhancing customer engagement and sales.</p>
                        </div>
                        <div className="ml-6">
                            <svg className="w-12 h-12 text-purple-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                        </div>
                    </div>
                </div>
            </div>
            
            <hr className="h-1 bg-gradient-to-r from-purple-700 to-blue-700 border-0 mt-20" />
        </div>
    );
}

export default Solutions;