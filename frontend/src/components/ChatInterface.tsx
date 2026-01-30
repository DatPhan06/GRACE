import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Film, Sparkles, Loader2 } from 'lucide-react';
import { sendMessage, type ChatResponse, type MovieRecommendation } from '@/lib/api';
import { cn } from '@/lib/utils'; // Assuming you have this utility from shadcn setup

interface Message {
    id: string;
    role: 'user' | 'ai';
    content: string;
    recommendations?: MovieRecommendation[];
}

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '1',
            role: 'ai',
            content: "Hello! I'm Grace. I can help you find the perfect movie. What are you in the mood for today?"
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const data: ChatResponse = await sendMessage(userMessage.content);

            const aiMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'ai',
                content: data.response,
                recommendations: data.recommendations
            };

            setMessages(prev => [...prev, aiMessage]);
        } catch (error) {
            console.error("Failed to send message", error);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'ai',
                content: "I'm sorry, I encountered an error while processing your request. Please try again."
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
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
                    <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-purple-700">GRACE</h1>
                    <p className="text-sm text-gray-500 font-medium">Generative Recommendation & Conversational Engine</p>
                </div>
            </header>

            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto mb-6 px-2 space-y-6 scrollbar-thin scrollbar-thumb-gray-200 scrollbar-track-transparent">
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={cn(
                            "flex gap-4 max-w-3xl",
                            msg.role === 'user' ? "ml-auto flex-row-reverse" : ""
                        )}
                    >
                        <div className={cn(
                            "w-10 h-10 rounded-full flex items-center justify-center shrink-0 shadow-sm",
                            msg.role === 'ai'
                                ? "bg-white border border-gray-100 text-blue-600"
                                : "bg-blue-600 text-white"
                        )}>
                            {msg.role === 'ai' ? <Bot size={20} /> : <User size={20} />}
                        </div>

                        <div className="space-y-4">
                            <div className={cn(
                                "p-4 rounded-2xl shadow-sm text-base leading-relaxed",
                                msg.role === 'ai'
                                    ? "bg-white/60 backdrop-blur-sm border border-white/50 text-gray-800 rounded-tl-none"
                                    : "bg-blue-600 text-white rounded-tr-none"
                            )}>
                                {msg.content}
                            </div>

                            {/* Recommendations Grid */}
                            {msg.recommendations && msg.recommendations.length > 0 && (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                                    {msg.recommendations.map((movie) => (
                                        <div
                                            key={movie.movieId}
                                            className="group relative overflow-hidden rounded-xl bg-white border border-gray-100 shadow-sm hover:shadow-md transition-all duration-300 hover:-translate-y-1 cursor-pointer"
                                        >
                                            <div className="p-4 flex gap-4">
                                                <div className="w-16 h-24 bg-gray-100 rounded-lg shrink-0 flex items-center justify-center text-gray-400">
                                                    <Film size={24} />
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <h3 className="font-semibold text-gray-900 truncate group-hover:text-blue-600 transition-colors">
                                                        {movie.title}
                                                    </h3>
                                                    {movie.year && (
                                                        <p className="text-sm text-gray-500 mt-1">{movie.year}</p>
                                                    )}
                                                    <div className="mt-2 flex items-center gap-1">
                                                        <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden">
                                                            <div
                                                                className="h-full bg-green-500 rounded-full"
                                                                style={{ width: `${(movie.score || 0) * 100}%` }}
                                                            />
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            {movie.plot && (
                                                <div className="absolute inset-0 bg-black/80 text-white p-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center text-center text-sm">
                                                    <p className="line-clamp-4">{movie.plot}</p>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {isLoading && (
                    <div className="flex gap-4">
                        <div className="w-10 h-10 rounded-full bg-white border border-gray-100 text-blue-600 flex items-center justify-center shrink-0 shadow-sm">
                            <Bot size={20} />
                        </div>
                        <div className="flex items-center gap-2 p-4 rounded-2xl rounded-tl-none bg-white/60 backdrop-blur-sm border border-white/50">
                            <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
                            <span className="text-sm text-gray-500 font-medium">Thinking...</span>
                        </div>
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
                        {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
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
