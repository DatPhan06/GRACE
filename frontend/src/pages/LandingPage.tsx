import { useState } from 'react';
import { MessageSquare, BarChart2, Menu } from 'lucide-react';
import ChatInterface from '@/components/ChatInterface';
import EvaluationPage from '@/pages/EvaluationPage';

export default function LandingPage() {
    const [activeTab, setActiveTab] = useState<'chat' | 'evaluation'>('chat');
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);

    const NavItem = ({ id, label, icon: Icon }: { id: 'chat' | 'evaluation', label: string, icon: any }) => (
        <button
            onClick={() => setActiveTab(id)}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors mb-1 ${activeTab === id
                    ? 'bg-blue-50 text-blue-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`}
        >
            <Icon size={20} />
            {isSidebarOpen && <span>{label}</span>}
        </button>
    );

    return (
        <div className="flex h-screen bg-gray-50 overflow-hidden">
            {/* Sidebar */}
            <aside
                className={`${isSidebarOpen ? 'w-64' : 'w-20'
                    } bg-white border-r border-gray-200 flex flex-col transition-all duration-300 shadow-sm z-10`}
            >
                {/* Logo / Header */}
                <div className="h-16 flex items-center px-6 border-b border-gray-100">
                    <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                        <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-lg">
                            G
                        </div>
                        {isSidebarOpen && <span className="font-bold text-xl text-gray-800">GRACE</span>}
                    </div>
                </div>

                {/* Navigation */}
                <nav className="flex-1 p-4 py-6">
                    <NavItem id="chat" label="Assistant" icon={MessageSquare} />
                    <NavItem id="evaluation" label="Evaluation" icon={BarChart2} />
                </nav>

                {/* Footer User Profile (Optional placeholder) */}
                <div className="p-4 border-t border-gray-100">
                    <div className="flex items-center space-x-3 px-2 py-2">
                        <div className="w-8 h-8 rounded-full bg-gray-200 flex-shrink-0"></div>
                        {isSidebarOpen && (
                            <div className="overflow-hidden">
                                <p className="text-sm font-medium text-gray-700 truncate">Admin User</p>
                                <p className="text-xs text-gray-500 truncate">admin@grace.ai</p>
                            </div>
                        )}
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
                <header className="h-16 bg-white border-b border-gray-200 flex items-center px-6 lg:hidden">
                    <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 -ml-2 text-gray-600">
                        <Menu size={24} />
                    </button>
                    <span className="ml-4 font-semibold text-gray-800">
                        {activeTab === 'chat' ? 'Chat Assistant' : 'Evaluation Dashboard'}
                    </span>
                </header>

                <div className="flex-1 overflow-auto p-4 sm:p-6 lg:p-8">
                    <div className="max-w-7xl mx-auto h-full">
                        {activeTab === 'chat' ? (
                            <ChatInterface />
                        ) : (
                            <EvaluationPage />
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
