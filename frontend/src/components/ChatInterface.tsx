import React, { useState, useRef, useEffect } from 'react';
import { Send, Film, Sparkles, Loader2, Check, Bot, User } from 'lucide-react';
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

interface NodeDef {
    id: AgentNode;
    label: string;
    short: string;
}

const NODE_DEFS: NodeDef[] = [
    { id: 'profiler',      label: 'Profiler Agent',    short: 'Profile'  },
    { id: 'orchestrator',  label: 'Orchestrator',       short: 'Plan'     },
    { id: 'retrieval',     label: 'Retrieval',          short: 'Retrieve' },
    { id: 'reranking',     label: 'Critic & Ranker',    short: 'Rank'     },
    { id: 'generation',    label: 'Generator',          short: 'Generate' },
];

interface NodeState {
    status: 'idle' | 'running' | 'done';
    message: string;
}

// ─── Agent progress strip ─────────────────────────────────────────────────────

function AgentProgress({ nodeStates }: { nodeStates: Record<AgentNode, NodeState> }) {
    console.log("AgentProgress rendering with nodeStates props:", JSON.parse(JSON.stringify(nodeStates)));
    const activeNode = NODE_DEFS.find(n => nodeStates[n.id].status === 'running');
    const doneCount = NODE_DEFS.filter(n => nodeStates[n.id].status === 'done').length;

    // Find the latest active or completed node to display its message
    const lastActiveOrDoneNode = [...NODE_DEFS].reverse().find(n => nodeStates[n.id].status !== 'idle');
    const displayMessage = lastActiveOrDoneNode ? nodeStates[lastActiveOrDoneNode.id].message : '';

    return (
        <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm w-72">
            {/* Header */}
            <div className="flex items-center gap-2 mb-4">
                <div className="flex gap-1">
                    {[0, 150, 300].map(d => (
                        <div key={d} className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: `${d}ms` }} />
                    ))}
                </div>
                <span className="text-xs font-medium text-gray-500">
                    {activeNode ? activeNode.label : 'Processing…'}
                </span>
            </div>

            {/* Step dots */}
            <div className="flex items-start gap-1">
                {NODE_DEFS.map((node, idx) => {
                    const s = nodeStates[node.id];
                    return (
                        <React.Fragment key={node.id}>
                            <div className="flex flex-col items-center gap-1.5 flex-1">
                                <div className={cn(
                                    'w-7 h-7 rounded-full flex items-center justify-center border-2 transition-all duration-300',
                                    s.status === 'done'
                                        ? 'bg-green-500 border-green-500'
                                        : s.status === 'running'
                                            ? 'bg-blue-500 border-blue-500 ring-4 ring-blue-100'
                                            : 'bg-white border-gray-200'
                                )}>
                                    {s.status === 'done'
                                        ? <Check size={11} className="text-white" />
                                        : s.status === 'running'
                                            ? <Loader2 size={11} className="text-white animate-spin" />
                                            : <span className="text-[10px] text-gray-400 font-medium">{idx + 1}</span>
                                    }
                                </div>
                                <span className={cn(
                                    'text-[9px] text-center leading-tight',
                                    s.status === 'done' ? 'text-green-600 font-medium'
                                        : s.status === 'running' ? 'text-blue-600 font-semibold'
                                            : 'text-gray-400'
                                )}>
                                    {node.short}
                                </span>
                            </div>
                            {idx < NODE_DEFS.length - 1 && (
                                <div className={`mt-3.5 h-0.5 flex-1 rounded-full transition-colors duration-300 ${
                                    s.status === 'done' ? 'bg-green-300' : 'bg-gray-100'
                                }`} />
                            )}
                        </React.Fragment>
                    );
                })}
            </div>

            {/* Progress bar */}
            <div className="mt-4 h-1 bg-gray-100 rounded-full overflow-hidden">
                <div
                    className="h-full bg-blue-500 rounded-full transition-all duration-500"
                    style={{ width: `${(doneCount / NODE_DEFS.length) * 100}%` }}
                />
            </div>

            {/* Active message */}
            {displayMessage && (
                <div className="mt-3 pt-2.5 border-t border-gray-100 text-[10px] text-gray-500 leading-normal animate-fade-in">
                    {displayMessage}
                </div>
            )}
        </div>
    );
}

// ─── Movie card ───────────────────────────────────────────────────────────────

function MovieCard({ movie }: { movie: MovieRecommendation }) {
    const [showPlot, setShowPlot] = useState(false);
    return (
        <div
            className="flex-shrink-0 w-40 bg-white border border-gray-200 rounded-xl overflow-hidden hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 cursor-pointer group"
            onClick={() => setShowPlot(v => !v)}
        >
            <div className="w-full h-24 bg-gray-100 flex items-center justify-center relative overflow-hidden">
                <Film size={26} className="text-gray-300" />
                {showPlot && movie.plot && (
                    <div className="absolute inset-0 bg-gray-900/90 flex items-center p-2.5">
                        <p className="text-white text-[9px] leading-relaxed line-clamp-6">{movie.plot}</p>
                    </div>
                )}
            </div>
            <div className="p-2.5">
                <h4 className="text-[11px] font-semibold text-gray-900 line-clamp-2 leading-tight group-hover:text-blue-600 transition-colors mb-1">
                    {movie.title}
                </h4>
                <div className="flex items-center justify-between">
                    {movie.year && <span className="text-[10px] text-gray-400">{movie.year}</span>}
                    {movie.imdbRating != null && (
                        <span className="text-[10px] text-amber-500 font-semibold">★ {movie.imdbRating.toFixed(1)}</span>
                    )}
                </div>
                {movie.similarity != null && (
                    <div className="mt-1.5 h-0.5 bg-gray-100 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-blue-400 rounded-full"
                            style={{ width: `${Math.min(movie.similarity * 500, 100)}%` }}
                        />
                    </div>
                )}
            </div>
        </div>
    );
}

// ─── Agent trace (collapsible) ─────────────────────────────────────────────────

function AgentTrace({ steps }: { steps: string[] }) {
    const [open, setOpen] = useState(false);
    return (
        <div className="mb-2">
            <button
                onClick={() => setOpen(v => !v)}
                className="flex items-center gap-1.5 text-[10px] text-gray-400 hover:text-gray-600 transition-colors select-none"
            >
                <Sparkles size={10} />
                <span>Agent trace ({steps.length} steps)</span>
                <span className={`transition-transform duration-200 ${open ? 'rotate-180' : ''}`}>▾</span>
            </button>
            {open && (
                <div className="mt-2 bg-gray-50 border border-gray-100 rounded-xl p-3 space-y-1.5">
                    {steps.map((step, idx) => (
                        <div key={idx} className="flex gap-2 text-[11px] text-gray-500 leading-relaxed">
                            <span className="text-gray-300 shrink-0 font-mono">{String(idx + 1).padStart(2, '0')}</span>
                            <span>{step}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

// ─── Initial state ────────────────────────────────────────────────────────────

const initialNodeStates = (): Record<AgentNode, NodeState> => ({
    profiler:     { status: 'idle', message: '' },
    orchestrator: { status: 'idle', message: '' },
    retrieval:    { status: 'idle', message: '' },
    reranking:    { status: 'idle', message: '' },
    generation:   { status: 'idle', message: '' },
});

// ─── Main component ───────────────────────────────────────────────────────────

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '1',
            role: 'ai',
            content: "Hi! I'm **GRACE** — your movie recommendation assistant. Tell me what you're in the mood for, and I'll find the perfect film for you.",
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

        const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);
        setNodeStates(initialNodeStates());

        const turns = messages.filter((m, i) => !(i === 0 && m.role === 'ai'));
        const history = turns.map(m => `${m.role === 'user' ? 'User' : 'GRACE'}: ${m.content}`).join('\n');
        const fullConversation = history ? `${history}\nUser: ${input}` : `User: ${input}`;

        try {
            console.log("streamChatMessages started");
            await streamChatMessages(fullConversation, event => {
                console.log("Stream event received:", event);
                try {
                    if (event.event === 'node') {
                        console.log(`Setting node state for ${event.node} to ${event.status}`);
                        setNodeStates(prev => {
                            const updated = {
                                ...prev,
                                [event.node]: { status: event.status, message: event.message },
                            };
                            console.log("Updated nodeStates:", updated);
                            return updated;
                        });
                    } else if (event.event === 'result') {
                        console.log("Result event received:", event);
                        setMessages(prev => [
                            ...prev,
                            {
                                id: (Date.now() + 1).toString(),
                                role: 'ai' as const,
                                content: event.response,
                                recommendations: event.recommendations,
                                agentTrace: event.agent_trace,
                            },
                        ]);
                        setIsLoading(false);
                    } else if (event.event === 'error') {
                        console.error("Error event from stream:", event.detail);
                        setMessages(prev => [
                            ...prev,
                            {
                                id: (Date.now() + 1).toString(),
                                role: 'ai' as const,
                                content: `Something went wrong: ${event.detail}`,
                            },
                        ]);
                        setIsLoading(false);
                    }
                } catch (handlerErr) {
                    console.error("Error in stream event handler:", handlerErr);
                }
            });
            console.log("streamChatMessages finished successfully");
        } catch (err) {
            console.error("streamChatMessages thrown error:", err);
            setMessages(prev => [
                ...prev,
                { id: (Date.now() + 1).toString(), role: 'ai', content: "Sorry, something went wrong. Please try again." },
            ]);
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-gray-50">

            {/* ── Chat area ── */}
            <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
                <div className="max-w-3xl mx-auto space-y-6">

                    {messages.map(msg => (
                        <div key={msg.id} className={cn('flex gap-3', msg.role === 'user' ? 'flex-row-reverse' : '')}>

                            {/* Avatar */}
                            <div className={cn(
                                'w-8 h-8 rounded-xl flex items-center justify-center shrink-0 mt-0.5',
                                msg.role === 'ai'
                                    ? 'bg-gray-900 text-white'
                                    : 'bg-blue-600 text-white'
                            )}>
                                {msg.role === 'ai' ? <Bot size={16} /> : <User size={16} />}
                            </div>

                            <div className={cn('flex flex-col gap-2', msg.role === 'user' ? 'items-end' : 'items-start', 'max-w-[80%]')}>

                                {/* Agent trace */}
                                {msg.role === 'ai' && msg.agentTrace && msg.agentTrace.length > 0 && (
                                    <AgentTrace steps={msg.agentTrace} />
                                )}

                                {/* Bubble */}
                                <div className={cn(
                                    'px-4 py-3 rounded-2xl text-sm leading-relaxed',
                                    msg.role === 'ai'
                                        ? 'bg-white border border-gray-200 text-gray-800 shadow-sm prose prose-sm prose-slate max-w-none rounded-tl-md'
                                        : 'bg-gray-900 text-white rounded-tr-md'
                                )}>
                                    {msg.role === 'ai'
                                        ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                                        : msg.content
                                    }
                                </div>

                                {/* Movie recommendations */}
                                {msg.recommendations && msg.recommendations.length > 0 && (
                                    <div className="w-full">
                                        <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-2 pl-1">
                                            Recommendations ({msg.recommendations.length})
                                        </p>
                                        <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-gray-200 scrollbar-track-transparent">
                                            {msg.recommendations.map(movie => (
                                                <MovieCard key={movie.movieId} movie={movie} />
                                            ))}
                                        </div>
                                    </div>
                                )}

                            </div>
                        </div>
                    ))}

                    {/* Agent progress while streaming */}
                    {isLoading && (
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-xl bg-gray-900 text-white flex items-center justify-center shrink-0 mt-0.5">
                                <Bot size={16} />
                            </div>
                            <AgentProgress nodeStates={nodeStates} />
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* ── Input area ── */}
            <div className="shrink-0 border-t border-gray-200 bg-white px-4 py-3">
                <div className="max-w-3xl mx-auto">
                    <form
                        onSubmit={handleSend}
                        className="flex items-center gap-2 bg-gray-50 border border-gray-200 rounded-2xl px-3 py-2 focus-within:border-blue-400 focus-within:bg-white transition-all"
                    >
                        <input
                            type="text"
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            placeholder="Tell me what kind of movie you're looking for…"
                            className="flex-1 bg-transparent px-2 py-1.5 text-sm text-gray-800 placeholder:text-gray-400 outline-none"
                            disabled={isLoading}
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className="w-8 h-8 flex items-center justify-center bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl transition-colors shrink-0"
                        >
                            {isLoading
                                ? <Loader2 size={15} className="animate-spin" />
                                : <Send size={15} />
                            }
                        </button>
                    </form>
                    <p className="text-center text-[10px] text-gray-400 mt-2">
                        Powered by the ARGOS multi-agent pipeline
                    </p>
                </div>
            </div>

        </div>
    );
}
