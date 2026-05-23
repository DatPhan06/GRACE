import { useState } from 'react';
import { MessageSquare, BarChart2, Layers, Film, ChevronLeft } from 'lucide-react';
import ChatInterface from '@/components/ChatInterface';
import EvaluationPage from '@/pages/EvaluationPage';
import StepEvaluationPage from '@/pages/StepEvaluationPage';

type TabId = 'chat' | 'evaluation' | 'step-evaluation';

const NAV_ITEMS: { id: TabId; label: string; icon: typeof MessageSquare; description: string }[] = [
    { id: 'chat',            label: 'Assistant',         icon: MessageSquare, description: 'Chat with ARGOS' },
    { id: 'evaluation',      label: 'Evaluation',         icon: BarChart2,     description: 'System metrics' },
    { id: 'step-evaluation', label: 'Step Optimization',  icon: Layers,        description: 'Pipeline stages' },
];

export default function LandingPage() {
    const [activeTab, setActiveTab] = useState<TabId>('chat');
    const [collapsed, setCollapsed] = useState(false);

    return (
        <div className="flex h-screen bg-gray-50 overflow-hidden">

            {/* ── Sidebar ── */}
            <aside className={`${collapsed ? 'w-16' : 'w-56'} shrink-0 bg-white border-r border-gray-200 flex flex-col transition-all duration-200`}>

                {/* Logo */}
                <div className="h-14 flex items-center justify-between px-3 border-b border-gray-100">
                    <div className="flex items-center gap-2.5 min-w-0">
                        <div className="w-8 h-8 bg-gray-900 rounded-xl flex items-center justify-center shrink-0">
                            <Film size={15} className="text-white" />
                        </div>
                        {!collapsed && (
                            <div className="min-w-0">
                                <p className="text-sm font-bold text-gray-900 tracking-tight leading-none">ARGOS</p>
                                <p className="text-[10px] text-gray-400 leading-none mt-0.5">Movie Recommender</p>
                            </div>
                        )}
                    </div>
                    {!collapsed && (
                        <button
                            onClick={() => setCollapsed(true)}
                            className="p-1 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
                        >
                            <ChevronLeft size={15} />
                        </button>
                    )}
                </div>

                {/* Nav */}
                <nav className="flex-1 p-2 space-y-0.5 pt-3">
                    {NAV_ITEMS.map(item => {
                        const isActive = activeTab === item.id;
                        return (
                            <button
                                key={item.id}
                                onClick={() => {
                                    setActiveTab(item.id);
                                    if (collapsed) setCollapsed(false);
                                }}
                                title={collapsed ? item.label : undefined}
                                className={`w-full flex items-center gap-3 px-2.5 py-2.5 rounded-xl transition-all text-sm group ${
                                    isActive
                                        ? 'bg-gray-900 text-white'
                                        : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                                }`}
                            >
                                <item.icon size={17} className="shrink-0" />
                                {!collapsed && (
                                    <div className="text-left min-w-0">
                                        <p className="font-medium leading-tight text-[13px]">{item.label}</p>
                                        <p className={`text-[10px] leading-tight mt-0.5 ${isActive ? 'text-gray-300' : 'text-gray-400'}`}>
                                            {item.description}
                                        </p>
                                    </div>
                                )}
                            </button>
                        );
                    })}
                </nav>

                {/* Expand button when collapsed */}
                {collapsed && (
                    <div className="p-2 border-t border-gray-100">
                        <button
                            onClick={() => setCollapsed(false)}
                            className="w-full p-2 rounded-xl text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors flex items-center justify-center"
                        >
                            <ChevronLeft size={15} className="rotate-180" />
                        </button>
                    </div>
                )}

                {/* Footer */}
                {!collapsed && (
                    <div className="p-3 border-t border-gray-100">
                        <p className="text-[10px] text-gray-400 text-center">ARGOS Pipeline · v1.0</p>
                    </div>
                )}
            </aside>

            {/* ── Main content ── */}
            <main className="flex-1 flex flex-col overflow-hidden min-w-0">

                {/* Content area — no padding for chat, scrollable for other pages */}
                <div className="flex-1 overflow-hidden">
                    {activeTab === 'chat' && (
                        <div className="h-full">
                            <ChatInterface />
                        </div>
                    )}
                    {activeTab === 'evaluation' && (
                        <div className="h-full overflow-auto">
                            <EvaluationPage />
                        </div>
                    )}
                    {activeTab === 'step-evaluation' && (
                        <div className="h-full overflow-auto">
                            <StepEvaluationPage />
                        </div>
                    )}
                </div>

            </main>
        </div>
    );
}
