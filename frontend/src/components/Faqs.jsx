import React, { useState } from 'react'

function Faqs() {
  const [openFaq, setOpenFaq] = useState(null);

  const faqs = [
    {
      question: "What problem does RetailVista solve?",
      answer: "It addresses retail theft prevention and enhances customer shopping experiences with smart solutions like real-time shoplifting detection and interactive navigation."
    },
    {
      question: "How does RetailVista improve store security?",
      answer: "The system uses computer vision for shoplifting detection and sends real-time alerts to administrators via an on-screen notification system."
    },
    {
      question: "What features help store owners analyze customer behavior?",
      answer: "RetailVista analyzes customer demographics and creates heat maps of store traffic to optimize product placement and marketing strategies."
    },
    {
      question: "How does RetailVista enhance the customer experience?",
      answer: "It offers an interactive store map for easy navigation and displays promotions to keep customers informed and engaged."
    },
    {
      question: "What technology powers RetailVista's modules?",
      answer: "The system integrates tools like Yolo, DeepSORT, and UTKFace for analysis and uses MappedIn for interactive store mapping."
    },
    {
      question: "Will this solution add more workload to the store and employees?",
      answer: "Our system is designed to enhance efficiency without adding extra workload to your store and employees. We can adjust sensitivity settings to match your preferences, ensuring that the system integrates smoothly with your current operations while maintaining employee productivity."
    }
  ];

  const toggleFaq = (index) => {
    setOpenFaq(openFaq === index ? null : index);
  };

  return (
    <div id="faqs" className="min-h-screen bg-black">
      <hr className="h-1 bg-gradient-to-r from-purple-700 to-blue-700 border-0" />
      
      <div className="container mx-auto px-4 py-16">
        <h2 className="text-4xl font-bold text-center text-white mb-12">
          Frequently Asked Questions
        </h2>
        
        <div className="max-w-3xl mx-auto space-y-6">
          {faqs.map((faq, index) => (
            <div key={index} className="bg-gradient-to-r from-purple-700/20 to-blue-700/20 rounded-lg">
              <button
                className="w-full p-6 flex justify-between items-center"
                onClick={() => toggleFaq(index)}
              >
                <h3 className="text-xl font-semibold text-white text-left">{faq.question}</h3>
                <svg
                  className={`w-6 h-6 text-white transform transition-transform duration-200 ${
                    openFaq === index ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>
              {openFaq === index && (
                <div className="px-6 pb-6">
                  <p className="text-gray-300">{faq.answer}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default Faqs