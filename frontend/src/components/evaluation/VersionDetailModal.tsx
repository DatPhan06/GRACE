import { BatchDetailResponse, BatchDetailItem } from '@/lib/api';

interface Props {
    detail: BatchDetailResponse;
    onClose: () => void;
    onRunNextStep?: (runId: number, batchId: number) => void;
    nextStepConfig?: any;
    loading?: boolean;
}

const stepColors: Record<string, { bg: string; border: string; text: string; badge: string }> = {
    summarization: { bg: 'bg-indigo-50', border: 'border-indigo-200', text: 'text-indigo-800', badge: 'bg-indigo-100 text-indigo-700' },
    retrieval: { bg: 'bg-purple-50', border: 'border-purple-200', text: 'text-purple-800', badge: 'bg-purple-100 text-purple-700' },
    reranking: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-800', badge: 'bg-green-100 text-green-700' },
};

const nextStepInfo: Record<string, { label: string; icon: string; color: string; hoverColor: string }> = {
    summarization: { label: 'Run Retrieval →', icon: '🔍', color: 'bg-purple-600', hoverColor: 'hover:bg-purple-700' },
    retrieval: { label: 'Run Reranking →', icon: '🏆', color: 'bg-green-600', hoverColor: 'hover:bg-green-700' },
};

function SummarizationItem({ item }: { item: BatchDetailItem }) {
    return (
        <div className="border border-gray-200 rounded-lg p-3 hover:shadow-sm transition-shadow">
            <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-mono bg-gray-100 px-2 py-0.5 rounded">{item.conv_id}</span>
            </div>
            <div className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed max-h-32 overflow-y-auto">
                {item.user_preferences || <span className="text-gray-400 italic">No preferences extracted</span>}
            </div>
        </div>
    );
}

function RetrievalItem({ item }: { item: BatchDetailItem }) {
    return (
        <div className="border border-gray-200 rounded-lg p-3 hover:shadow-sm transition-shadow">
            <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono bg-gray-100 px-2 py-0.5 rounded">{item.conv_id}</span>
                <span className="text-xs text-gray-500">{item.candidate_count} candidates</span>
            </div>
            <div className="flex gap-3 text-xs text-gray-600">
                {item.semantic_count != null && <span className="bg-blue-50 px-2 py-0.5 rounded">Semantic: {item.semantic_count}</span>}
                {item.content_count != null && <span className="bg-orange-50 px-2 py-0.5 rounded">Content: {item.content_count}</span>}
                {item.collab_count != null && <span className="bg-teal-50 px-2 py-0.5 rounded">Collab: {item.collab_count}</span>}
            </div>
            {item.candidates && item.candidates.length > 0 && (
                <details className="mt-2">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                        Show candidates
                    </summary>
                    <div className="mt-1 max-h-40 overflow-y-auto text-xs font-mono bg-gray-50 p-2 rounded">
                        {item.candidates.map((c, i) => (
                            <div key={i} className="py-0.5 border-b border-gray-100 last:border-0">
                                {typeof c === 'string' ? c : (c?.title || JSON.stringify(c))}
                            </div>
                        ))}
                    </div>
                </details>
            )}
        </div>
    );
}

function RerankingItem({ item }: { item: BatchDetailItem }) {
    return (
        <div className="border border-gray-200 rounded-lg p-3 hover:shadow-sm transition-shadow">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <span className="text-xs font-mono bg-gray-100 px-2 py-0.5 rounded">{item.conv_id}</span>
                    {item.model_used && <span className="text-xs bg-gray-200 px-1.5 py-0.5 rounded capitalize">{item.model_used}</span>}
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500">{item.reranked_count} items</span>
                    {item.recall != null && (
                        <span className={`text-xs font-semibold px-2 py-0.5 rounded ${item.recall > 0 ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-600'}`}>
                            Recall: {item.recall.toFixed(3)}
                        </span>
                    )}
                </div>
            </div>
            {item.reranked_candidates && item.reranked_candidates.length > 0 && (
                <details className="mt-1">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                        Show reranked list
                    </summary>
                    <div className="mt-1 max-h-40 overflow-y-auto text-xs font-mono bg-gray-50 p-2 rounded">
                        {item.reranked_candidates.map((c, i) => (
                            <div key={i} className="py-0.5 border-b border-gray-100 last:border-0 flex gap-2">
                                <span className="text-gray-400 w-5 text-right">{i + 1}.</span>
                                <span>{typeof c === 'string' ? c : (c?.title || JSON.stringify(c))}</span>
                            </div>
                        ))}
                    </div>
                </details>
            )}
        </div>
    );
}

export default function VersionDetailModal({ detail, onClose, onRunNextStep, nextStepConfig, loading }: Props) {
    const colors = stepColors[detail.step_type] || stepColors.summarization;
    const nextStep = nextStepInfo[detail.step_type];

    // Compute summary stats for reranking
    const avgRecall = detail.step_type === 'reranking'
        ? detail.items.filter(i => i.recall != null).reduce((sum, i) => sum + (i.recall || 0), 0) / (detail.items.filter(i => i.recall != null).length || 1)
        : null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
            <div className="bg-white rounded-xl shadow-2xl w-[90vw] max-w-3xl max-h-[85vh] flex flex-col" onClick={e => e.stopPropagation()}>
                {/* Header */}
                <div className={`${colors.bg} ${colors.border} border-b px-6 py-4 rounded-t-xl flex justify-between items-start`}>
                    <div>
                        <div className="flex items-center gap-3 mb-1">
                            <h2 className={`text-xl font-bold ${colors.text} capitalize`}>{detail.step_type} v{detail.version}</h2>
                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${colors.badge}`}>{detail.status}</span>
                        </div>
                        <div className="text-xs text-gray-500 space-x-3">
                            <span>Batch #{detail.id}</span>
                            <span>Run #{detail.run_id}</span>
                            <span>{new Date(detail.created_at).toLocaleString()}</span>
                        </div>
                        {/* Config */}
                        {detail.config && Object.keys(detail.config).length > 0 && (
                            <div className="mt-2 flex gap-2 flex-wrap">
                                {Object.entries(detail.config).map(([k, v]) => (
                                    <span key={k} className="text-xs bg-white/70 px-2 py-0.5 rounded border border-gray-200">
                                        {k}: <strong>{String(v)}</strong>
                                    </span>
                                ))}
                            </div>
                        )}
                        {/* Avg Recall for reranking */}
                        {avgRecall != null && (
                            <div className="mt-2 text-sm font-semibold text-green-700">
                                Avg Recall: {avgRecall.toFixed(4)}
                            </div>
                        )}
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl leading-none p-1">&times;</button>
                </div>

                {/* Body */}
                <div className="flex-1 overflow-y-auto px-6 py-4">
                    <div className="text-sm text-gray-500 mb-3">{detail.items.length} conversations</div>
                    <div className="space-y-3">
                        {detail.items.map((item, idx) => (
                            <div key={idx}>
                                {detail.step_type === 'summarization' && <SummarizationItem item={item} />}
                                {detail.step_type === 'retrieval' && <RetrievalItem item={item} />}
                                {detail.step_type === 'reranking' && <RerankingItem item={item} />}
                            </div>
                        ))}
                        {detail.items.length === 0 && (
                            <div className="text-center text-gray-400 py-8">No detail data available for this batch.</div>
                        )}
                    </div>
                </div>

                {/* Footer with Next Step Action */}
                <div className="border-t border-gray-200 px-6 py-3 flex justify-between items-center">
                    <button onClick={onClose} className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors">
                        Close
                    </button>

                    {nextStep && onRunNextStep && (
                        <div className="flex items-center gap-3">
                            {/* Retrieval params when going from Summarization → Retrieval */}
                            {detail.step_type === 'summarization' && nextStepConfig && (
                                <div className="flex items-center gap-2">
                                    <label className="text-xs text-gray-500">N Candidates:</label>
                                    <input
                                        type="number"
                                        value={nextStepConfig.nSample}
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => nextStepConfig.setNSample(Number(e.target.value))}
                                        className="w-20 text-sm border border-gray-300 rounded px-2 py-1"
                                    />
                                </div>
                            )}

                            {/* Reranking params when going from Retrieval → Reranking */}
                            {detail.step_type === 'retrieval' && nextStepConfig && (
                                <div className="flex items-center gap-2">
                                    <label className="text-xs text-gray-500">Top K:</label>
                                    <input
                                        type="number"
                                        value={nextStepConfig.topK}
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => nextStepConfig.setTopK(Number(e.target.value))}
                                        className="w-16 text-sm border border-gray-300 rounded px-2 py-1"
                                    />
                                    <label className="text-xs text-gray-500">Model:</label>
                                    <select
                                        value={nextStepConfig.model}
                                        onChange={(e: React.ChangeEvent<HTMLSelectElement>) => nextStepConfig.setModel(e.target.value)}
                                        className="text-sm border border-gray-300 rounded px-2 py-1"
                                    >
                                        <option value="cohere">Cohere</option>
                                        <option value="llm">LLM</option>
                                    </select>
                                </div>
                            )}

                            <button
                                onClick={() => onRunNextStep(detail.run_id, detail.id)}
                                disabled={loading}
                                className={`flex items-center gap-2 px-5 py-2 text-sm text-white rounded-lg ${nextStep.color} ${nextStep.hoverColor} disabled:opacity-50 transition-colors font-medium shadow-sm`}
                            >
                                <span>{nextStep.icon}</span>
                                {loading ? 'Running...' : nextStep.label}
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
