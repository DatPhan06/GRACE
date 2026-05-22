import React, { useState, useRef, useEffect } from 'react';
import { flushSync } from 'react-dom';
import { Send, Bot, User, Film, Sparkles, Loader2, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
    streamChatMessages,
    type MovieRecommendation,
    type AgentNode,
} from '@/lib/api';
import { cn } from '@/lib/utils';

interface Message {
    id: string;
    role: 'user' | 'ai';
    content: string;
    recommendations?: MovieRecommendation[];
    agentTrace?: string[];
}

// ─── Node definitions ────────────────────────────────────────────────────────
interface NodeDef {
    id: AgentNode;
    icon: React.ReactNode;
    label: string;
    runningLabel: string;
}

const NODE_DEFS: NodeDef[] = [
    {
        id: 'profiler',
        icon: <Sparkles size={15} />,
        label: 'Profiler Agent',
        runningLabel: 'Analyzing your preferences...',
    },
    {
        id: 'orchestrator',
        icon: <Bot size={15} />,
        label: 'Orchestrator',
        runningLabel: 'Planning retrieval strategy...',
    },
    {
        id: 'retrieval',
        icon: <Bot size={15} />,
        label: 'Retrieval Agents',
        runningLabel: 'Launching parallel retrieval streams...',
    },
    {
        id: 'reranking',
        icon: <Film size={15} />,
        label: 'Critic & Ranker',
        runningLabel: 'Filtering & ranking candidates...',
    },
    {
        id: 'generation',
        icon: <Sparkles size={15} />,
        label: 'Generator',
        runningLabel: 'Composing your response...',
    },
];

// ─── Live Node Tracer component ───────────────────────────────────────────────
interface NodeState {
    status: 'idle' | 'running' | 'done';
    message: string;
}

function LiveNodeTracer({ nodeStates }: { nodeStates: Record<AgentNode, NodeState> }) {
    return (
        <div className="flex flex-col gap-3 p-4 rounded-2xl rounded-tl-none bg-white/70 backdrop-blur-sm border border-white/50 shadow-sm w-80">
            <div className="flex items-center gap-2 pb-2 border-b border-gray-100">
                <Loader2 size={15} className="animate-spin text-blue-600" />
                <span className="text-sm font-semibold text-gray-700">Agents are collaborating...</span>
            </div>
            {NODE_DEFS.map((node) => {
                const state = nodeStates[node.id];
                const isRunning = state.status === 'running';
                const isDone = state.status === 'done';

                return (
                    <div
                        key={node.id}
                        className={cn(
                            'flex items-start gap-3 transition-all duration-400',
                            state.status === 'idle' ? 'opacity-25' : 'opacity-100'
                        )}
                    >
                        {/* Status icon / circle */}
                        <div className={cn(
                            'mt-0.5 w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 border transition-all duration-300',
                            isDone
                                ? 'bg-green-500 border-green-500 text-white'
                                : isRunning
                                    ? 'bg-blue-100 border-blue-300 text-blue-600 animate-pulse'
                                    : 'bg-gray-50 border-gray-200 text-gray-300'
                        )}>
                            {isDone ? <Check size={12} /> : node.icon}
                        </div>

                        <div className="flex-1 min-w-0">
                            <p className={cn(
                                'text-xs font-semibold',
                                isDone ? 'text-green-700'
                                    : isRunning ? 'text-blue-700'
                                        : 'text-gray-400'
                            )}>
                                {node.label}
                            </p>
                            {(isRunning || isDone) && (
                                <p className={cn(
                                    'text-[11px] mt-0.5 leading-relaxed',
                                    isDone ? 'text-gray-400' : 'text-blue-500 italic'
                                )}>
                                    {isDone ? state.message : node.runningLabel}
                                </p>
                            )}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

// ─── Main Component ───────────────────────────────────────────────────────────
const initialNodeStates = (): Record<AgentNode, NodeState> => ({
    profiler: { status: 'idle', message: '' },
    orchestrator: { status: 'idle', message: '' },
    retrieval: { status: 'idle', message: '' },
    reranking: { status: 'idle', message: '' },
    generation: { status: 'idle', message: '' },
});

// Helper to get icon for the post-response agent trace list
const getStepIcon = (step: string) => {
    if (step.includes('Profiler')) return <Sparkles size={13} className="text-purple-500" />;
    if (step.includes('Orchestrator')) return <Bot size={13} className="text-blue-500" />;
    if (step.includes('Graph')) return <User size={13} className="text-green-500" />;
    if (step.includes('Critic') || step.includes('Ranker')) return <Film size={13} className="text-red-500" />;
    return <Loader2 size={13} className="text-gray-400" />;
};

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '1',
            role: 'ai',
            content: "Hello! I'm Grace. I can help you find the perfect movie. What are you in the mood for today?",
        },
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [nodeStates, setNodeStates] = useState<Record<AgentNode, NodeState>>(initialNodeStates());
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading, nodeStates]);

    const handleSend = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input,
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);
        setNodeStates(initialNodeStates());

        // Build full conversation history (skip the static initial greeting)
        const turns = messages.filter((m, i) => !(i === 0 && m.role === 'ai'));
        const history = turns
            .map((m) => `${m.role === 'user' ? 'User' : 'GRACE'}: ${m.content}`)
            .join('\n');
        const fullConversation = history
            ? `${history}\nUser: ${input}`
            : `User: ${input}`;

        try {
            await streamChatMessages(fullConversation, (event) => {
                if (event.event === 'node') {
                    // flushSync forces an immediate render for each node transition
                    // so the animation is visible even when events arrive in rapid succession
                    flushSync(() => {
                        setNodeStates((prev) => ({
                            ...prev,
                            [event.node]: {
                                status: event.status,
                                message: event.message,
                            },
                        }));
                    });
                } else if (event.event === 'result') {
                    const aiMessage: Message = {
                        id: (Date.now() + 1).toString(),
                        role: 'ai',
                        content: event.response,
                        recommendations: event.recommendations,
                        agentTrace: event.agent_trace,
                    };
                    setMessages((prev) => [...prev, aiMessage]);
                } else if (event.event === 'error') {
                    const errorMessage: Message = {
                        id: (Date.now() + 1).toString(),
                        role: 'ai',
                        content: `Sorry, an error occurred: ${event.detail}`,
                    };
                    setMessages((prev) => [...prev, errorMessage]);
                }
            });
        } catch (error) {
            console.error('Stream error', error);
            setMessages((prev) => [
                ...prev,
                {
                    id: (Date.now() + 1).toString(),
                    role: 'ai',
                    content: "I'm sorry, I encountered an error. Please try again.",
                },
            ]);
        } finally {
            setIsLoading(false);
            // nodeStates is intentionally NOT reset here — LiveNodeTracer is already
            // hidden when isLoading=false. It will be reset on the next request.
        }
    };

    return (
        <div className="flex flex-col h-screen max-w-5xl mx-auto p-4 md:p-6 lg:p-8">
            {/* Header */}
            <header className="flex items-center gap-3 mb-6 p-4 rounded-2xl bg-white/40 backdrop-blur-md border border-white/20 shadow-sm">
                <div className="p-3 bg-gradient-to-tr from-blue-600 to-purple-600 rounded-xl shadow-lg">
                    <Sparkles className="w-6 h-6 text-white" />
                </div>
                <div>
                    <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-purple-700">
                        GRACE
                    </h1>
                    <p className="text-sm text-gray-500 font-medium">
                        Generative Recommendation & Conversational Engine
                    </p>
                </div>
            </header>

            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto mb-6 px-2 space-y-6 scrollbar-thin scrollbar-thumb-gray-200 scrollbar-track-transparent">
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={cn(
                            'flex gap-4 max-w-4xl',
                            msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''
                        )}
                    >
                        <div
                            className={cn(
                                'w-10 h-10 rounded-full flex items-center justify-center shrink-0 shadow-sm',
                                msg.role === 'ai'
                                    ? 'bg-white border border-gray-100 text-blue-600'
                                    : 'bg-blue-600 text-white'
                            )}
                        >
                            {msg.role === 'ai' ? <Bot size={20} /> : <User size={20} />}
                        </div>

                        <div className="flex-1 space-y-3 max-w-[85%]">
                            {/* Collapsible Agent Trace */}
                            {msg.role === 'ai' && msg.agentTrace && msg.agentTrace.length > 0 && (
                                <details className="group">
                                    <summary className="list-none cursor-pointer flex items-center gap-2 text-[10px] uppercase tracking-wider font-bold text-gray-400 hover:text-blue-500 transition-colors select-none">
                                        <div className="flex -space-x-1">
                                            <div className="w-4 h-4 rounded-full bg-purple-100 border border-white flex items-center justify-center">
                                                <Sparkles size={7} />
                                            </div>
                                            <div className="w-4 h-4 rounded-full bg-blue-100 border border-white flex items-center justify-center">
                                                <Bot size={7} />
                                            </div>
                                            <div className="w-4 h-4 rounded-full bg-red-100 border border-white flex items-center justify-center">
                                                <Film size={7} />
                                            </div>
                                        </div>
                                        Agent Reflection Trace
                                        <div className="h-[1px] flex-1 bg-gray-100 group-open:bg-blue-100 transition-colors" />
                                    </summary>
                                    <div className="mt-2 p-3 rounded-xl bg-gray-50/70 border border-gray-100 space-y-2">
                                        {msg.agentTrace.map((step, idx) => (
                                            <div key={idx} className="flex gap-3 text-xs leading-relaxed text-gray-600">
                                                <div className="mt-0.5 shrink-0">{getStepIcon(step)}</div>
                                                <div className="flex-1 italic font-medium">{step}</div>
                                            </div>
                                        ))}
                                    </div>
                                </details>
                            )}

                            {/* Message Bubble */}
                            <div
                                className={cn(
                                    'p-4 rounded-2xl shadow-sm text-base leading-relaxed overflow-hidden',
                                    msg.role === 'ai'
                                        ? 'bg-white/60 backdrop-blur-sm border border-white/50 text-gray-800 rounded-tl-none prose prose-slate max-w-none'
                                        : 'bg-blue-600 text-white rounded-tr-none'
                                )}
                            >
                                {msg.role === 'ai' ? (
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                                ) : (
                                    msg.content
                                )}
                            </div>

                            {/* Recommendations Grid */}
                            {msg.recommendations && msg.recommendations.length > 0 && (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
                                    {msg.recommendations.map((movie) => (
                                        <div
                                            key={movie.movieId}
                                            className="group relative overflow-hidden rounded-xl bg-white border border-gray-100 shadow-sm hover:shadow-md transition-all duration-300 hover:-translate-y-1 cursor-pointer"
                                        >
                                            <div className="p-4 flex gap-4">
                                                <div className="w-14 h-20 bg-gray-100 rounded-lg shrink-0 flex items-center justify-center text-gray-400">
                                                    <Film size={22} />
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <h3 className="font-semibold text-gray-900 truncate group-hover:text-blue-600 transition-colors text-sm">
                                                        {movie.title}
                                                    </h3>
                                                    <div className="flex items-center gap-2 mt-0.5">
                                                        {movie.year && (
                                                            <p className="text-xs text-gray-500">{movie.year}</p>
                                                        )}
                                                        {movie.imdbRating != null && (
                                                            <p className="text-xs text-yellow-600 font-semibold">★ {movie.imdbRating.toFixed(1)}</p>
                                                        )}
                                                    </div>
                                                    {movie.similarity != null && (
                                                        <div className="mt-2">
                                                            <div className="h-1 w-full bg-gray-100 rounded-full overflow-hidden">
                                                                <div
                                                                    className="h-full bg-gradient-to-r from-blue-400 to-purple-500 rounded-full"
                                                                    style={{ width: `${Math.min(movie.similarity * 500, 100)}%` }}
                                                                />
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                            {movie.plot && (
                                                <div className="absolute inset-0 bg-black/80 text-white p-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center text-center text-xs">
                                                    <p className="line-clamp-5">{movie.plot}</p>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {/* Live Node Tracer — shown while streaming */}
                {isLoading && (
                    <div className="flex gap-4">
                        <div className="w-10 h-10 rounded-full bg-white border border-gray-100 text-blue-600 flex items-center justify-center shrink-0 shadow-sm">
                            <Bot size={20} />
                        </div>
                        <LiveNodeTracer nodeStates={nodeStates} />
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="relative">
                <form
                    onSubmit={handleSend}
                    className="flex items-center gap-2 p-2 bg-white rounded-2xl border border-gray-200 shadow-lg focus-within:ring-2 focus-within:ring-blue-500/20 focus-within:border-blue-500 transition-all"
                >
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about movies (e.g., 'I love Sci-Fi movies like Interstellar')..."
                        className="flex-1 px-4 py-3 bg-transparent outline-none text-gray-800 placeholder:text-gray-400"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg active:scale-95"
                    >
                        {isLoading ? (
                            <Loader2 size={20} className="animate-spin" />
                        ) : (
                            <Send size={20} />
                        )}
                    </button>
                </form>
                <p className="text-center text-xs text-gray-400 mt-3">
                    Grace may make mistakes. Please verify important information.
                </p>
            </div>

            {/* Background Decorations */}
            <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-400/20 rounded-full blur-3xl mix-blend-multiply animate-blob" />
                <div className="absolute top-0 right-1/4 w-96 h-96 bg-purple-400/20 rounded-full blur-3xl mix-blend-multiply animate-blob animation-delay-2000" />
                <div className="absolute bottom-0 left-1/3 w-96 h-96 bg-pink-400/20 rounded-full blur-3xl mix-blend-multiply animate-blob animation-delay-4000" />
            </div>
        </div>
    );
}
