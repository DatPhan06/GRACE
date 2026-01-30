import React from 'react';

export default function LandingPage() {
    return (
        <div className="flex min-h-screen flex-col items-center justify-center bg-gray-50 text-gray-900">
            <h1 className="text-4xl font-bold mb-4 text-blue-600">GRACE Frontend</h1>
            <p className="text-lg text-gray-700 max-w-2xl text-center">
                Welcome to your new React 19 + Vite app with Tailwind CSS v4 and Shadcn/UI support.
            </p>
            <div className="mt-8 flex gap-4">
                <button className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition">
                    Get Started
                </button>
                <button className="px-6 py-2 bg-white border border-gray-300 rounded-md hover:bg-gray-100 transition">
                    Documentation
                </button>
            </div>
        </div>
    );
}
