import { useState, useEffect, useCallback } from 'react';
import { initializeRun, runProfilerStep, runRetrievalStep, runRerankingStep, getStepsByType, getBatchDetail, getEvaluationRuns, getEvaluationInfo, getRunConversations, BatchStepExecutionResponse, BatchDetailResponse, EvaluationRunResponse, ConversationLogItem } from '@/lib/api';
import VersionDetailModal from '@/components/evaluation/VersionDetailModal';

// ─── Shared UI components ────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const VersionCard = ({ batch, onClick, children }: { batch: BatchStepExecutionResponse; onClick: () => void; children?: any }) => (
    <div
        onClick={onClick}
        className="p-4 bg-white border border-gray-200 rounded-2xl cursor-pointer hover:border-gray-300 hover:shadow-sm transition-all group"
    >
        <div className="flex justify-between items-start mb-2">
            <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-gray-900">
                    {batch.name ? batch.name : `v${batch.version}`}
                </span>
                {batch.name && <span className="text-xs text-gray-400">v{batch.version}</span>}
            </div>
            <span className="text-xs text-gray-400 group-hover:text-gray-600 transition-colors">{new Date(batch.created_at).toLocaleString()}</span>
        </div>
        <div className="flex justify-between items-center mb-2">
            <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 border border-gray-200 text-gray-600">Run #{batch.run_id}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full capitalize font-medium ${batch.status === 'completed' ? 'bg-green-100 text-green-700' : batch.status === 'failed' ? 'bg-red-100 text-red-600' : 'bg-amber-100 text-amber-700'}`}>
                {batch.status}
            </span>
        </div>
        {children && <div className="mt-2 pt-2 border-t border-gray-100">{children}</div>}
    </div>
);

const ModalBackdrop = ({ title, onClose, children, onSubmit, submitLabel, submitDisabled }: {
    title: string; onClose: () => void; children: React.ReactNode;
    onSubmit: () => void; submitLabel: string; submitDisabled?: boolean;
}) => (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
        <div className="bg-white rounded-2xl shadow-xl w-[90vw] max-w-lg flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center">
                <h3 className="text-base font-semibold text-gray-800">{title}</h3>
                <button onClick={onClose} className="w-8 h-8 rounded-xl flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors">&times;</button>
            </div>
            <div className="px-6 py-5 space-y-4">
                {children}
            </div>
            <div className="px-6 py-4 border-t border-gray-100 flex justify-end gap-3 bg-gray-50/50 rounded-b-2xl">
                <button onClick={onClose} className="px-4 py-2 text-sm bg-white border border-gray-200 hover:bg-gray-50 rounded-xl transition-colors text-gray-600">Cancel</button>
                <button onClick={(e) => { e.preventDefault(); onSubmit(); }} disabled={submitDisabled} className="px-6 py-2 text-sm text-white rounded-xl font-medium bg-gray-900 hover:bg-gray-800 disabled:opacity-40 transition-colors shadow-sm">
                    {submitLabel}
                </button>
            </div>
        </div>
    </div>
);

const EmptyState = ({ icon, message }: { icon: string; message: string }) => (
    <div className="text-center py-16">
        <div className="text-5xl mb-3 opacity-40">{icon}</div>
        <p className="text-gray-400 text-sm">{message}</p>
    </div>
);

// ─── Vertical Pipeline Diagram (right sidebar) ────────────────────────────────

function VerticalPipelineStep({
    step, icon, title, subtitle, color, active, onClick,
}: {
    step: string; icon: string; title: string; subtitle: string;
    color: { ring: string; bg: string; text: string; badge: string };
    active: boolean; onClick: () => void;
}) {
    return (
        <button
            onClick={onClick}
            className={`w-full text-left rounded-xl px-3 py-3 border-2 transition-all ${active ? `${color.ring} ${color.bg} shadow-sm` : 'border-gray-100 bg-white hover:border-gray-200 hover:shadow-sm'}`}
        >
            <div className="flex items-center gap-3">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-base shrink-0 ${active ? color.badge : 'bg-gray-100'}`}>
                    {icon}
                </div>
                <div className="min-w-0">
                    <div className={`text-xs font-bold uppercase tracking-wider mb-0.5 ${active ? color.text : 'text-gray-400'}`}>{step}</div>
                    <div className={`text-sm font-semibold leading-tight ${active ? color.text : 'text-gray-700'}`}>{title}</div>
                    <div className="text-xs text-gray-400 mt-0.5">{subtitle}</div>
                </div>
            </div>
        </button>
    );
}

function PipelineConnector({ active }: { active: boolean }) {
    return (
        <div className="flex justify-center py-1">
            <div className={`w-0.5 h-5 rounded-full transition-colors ${active ? 'bg-gray-400' : 'bg-gray-200'}`} />
        </div>
    );
}

function OrchestratorDetail() {
    return (
        <div className="mt-2 ml-11 mr-1 border border-gray-200 bg-gray-50 rounded-lg px-3 py-2.5 space-y-1.5">
            <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Parallel streams</div>
            <div className="flex gap-1">
                <span className="text-[9px] bg-white text-gray-600 px-1.5 py-0.5 rounded border border-gray-200">Semantic</span>
                <span className="text-[9px] bg-white text-gray-600 px-1.5 py-0.5 rounded border border-gray-200">Content</span>
                <span className="text-[9px] bg-white text-gray-600 px-1.5 py-0.5 rounded border border-gray-200">Graph</span>
            </div>
            <div className="text-[9px] text-gray-400 pl-0.5">↓ WRRF(w_sem, w_con, w_col)</div>
            <div className="text-[10px] bg-white border border-gray-200 rounded px-2 py-1 text-gray-700 font-medium">Critic Agent</div>
            <div className="border border-dashed border-gray-300 bg-white rounded px-2 py-1">
                <div className="text-[9px] text-gray-500 font-semibold">if |M| &lt; τ</div>
                <div className="text-[9px] text-gray-600">Relaxation → re-retrieve</div>
            </div>
            <div className="text-[9px] text-gray-400 pl-0.5">↓ Top-N candidates</div>
        </div>
    );
}

// ─── Tab definitions ──────────────────────────────────────────────────────────

type TabId = 'init' | 'profiler' | 'orchestrator' | 'reranker';

const TABS: { id: TabId; step: string; icon: string; title: string; subtitle: string; color: { ring: string; bg: string; text: string; badge: string } }[] = [
    {
        id: 'init',
        step: 'Step 1',
        icon: '🚀',
        title: 'Evaluation Runs',
        subtitle: 'Dataset samples',
        color: { ring: 'border-blue-400', bg: 'bg-blue-50/60', text: 'text-blue-700', badge: 'bg-blue-100' },
    },
    {
        id: 'profiler',
        step: 'Step 2',
        icon: '🤖',
        title: 'Profiler Agent',
        subtitle: 'Extract preferences',
        color: { ring: 'border-indigo-400', bg: 'bg-indigo-50/60', text: 'text-indigo-700', badge: 'bg-indigo-100' },
    },
    {
        id: 'orchestrator',
        step: 'Step 3',
        icon: '🎛️',
        title: 'Orchestrator Agent',
        subtitle: 'Retrieve & filter',
        color: { ring: 'border-purple-400', bg: 'bg-purple-50/60', text: 'text-purple-700', badge: 'bg-purple-100' },
    },
    {
        id: 'reranker',
        step: 'Step 4',
        icon: '🎯',
        title: 'Reranker',
        subtitle: 'Recall@K',
        color: { ring: 'border-green-400', bg: 'bg-green-50/60', text: 'text-green-700', badge: 'bg-green-100' },
    },
];

// ─── Main Page ─────────────────────────────────────────────────────────────────

export default function StepEvaluationPage() {
    const [activeTab, setActiveTab] = useState<TabId>('init');
    const [loading, setLoading] = useState(false);

    // Params
    const [dataset, setDataset] = useState<"inspired" | "redial">("redial");
    const [samplePercent, setSamplePercent] = useState(20);
    const [datasetSizes, setDatasetSizes] = useState<Record<string, number>>({});
    const [nSample, setNSample] = useState(100);
    const [topK, setTopK] = useState(10);
    const [model, setModel] = useState<"llm" | "cohere">("cohere");

    // Data
    const [allRuns, setAllRuns] = useState<EvaluationRunResponse[]>([]);
    const [summBatches, setSummBatches] = useState<BatchStepExecutionResponse[]>([]);
    const [retrievalBatches, setRetrievalBatches] = useState<BatchStepExecutionResponse[]>([]);
    const [rerankBatches, setRerankBatches] = useState<BatchStepExecutionResponse[]>([]);

    // Selection
    const [selectedRetrievalBatch, setSelectedRetrievalBatch] = useState<number | undefined>(undefined);

    // Detail Modal
    const [showDetailModal, setShowDetailModal] = useState(false);
    const [batchDetail, setBatchDetail] = useState<BatchDetailResponse | null>(null);
    const [detailLoading, setDetailLoading] = useState(false);

    // New Version Modal
    const [showNewVersionModal, setShowNewVersionModal] = useState<"summ" | "retrieval" | "rerank" | null>(null);
    const [newVersionName, setNewVersionName] = useState('');
    const [selectedRunId, setSelectedRunId] = useState<number | undefined>(undefined);
    const [selectedSummBatch, setSelectedSummBatch] = useState<number | undefined>(undefined);

    // Init Run Modal
    const [showRunModal, setShowRunModal] = useState(false);

    // Run Detail Modal
    const [showRunDetailModal, setShowRunDetailModal] = useState(false);
    const [runConversations, setRunConversations] = useState<ConversationLogItem[]>([]);
    const [runDetailLoading, setRunDetailLoading] = useState(false);
    const [selectedRun, setSelectedRun] = useState<EvaluationRunResponse | null>(null);

    // --- Data Loading ---
    const loadRuns = useCallback(async () => {
        try { setAllRuns(await getEvaluationRuns(0, 100)); } catch (e) { console.error(e); }
    }, []);

    const loadSummBatches = useCallback(async () => {
        try { setSummBatches(await getStepsByType('summarization')); } catch (e) { console.error(e); }
    }, []);

    const loadRetrievalBatches = useCallback(async () => {
        try { setRetrievalBatches(await getStepsByType('retrieval')); } catch (e) { console.error(e); }
    }, []);

    const loadRerankBatches = useCallback(async () => {
        try { setRerankBatches(await getStepsByType('reranking')); } catch (e) { console.error(e); }
    }, []);

    useEffect(() => {
        loadRuns();
        loadSummBatches();
        loadRetrievalBatches();
        loadRerankBatches();
        getEvaluationInfo().then(info => setDatasetSizes(info.dataset_sizes)).catch(() => {});
    }, [loadRuns, loadSummBatches, loadRetrievalBatches, loadRerankBatches]);

    const stepRunIds = new Set([
        ...summBatches.map(b => b.run_id),
        ...retrievalBatches.map(b => b.run_id),
        ...rerankBatches.map(b => b.run_id),
    ]);
    const stepRuns = allRuns.filter(r => r.status === 'initialized' || stepRunIds.has(r.id));

    // Polling
    useEffect(() => {
        const hasRunning =
            allRuns.some(r => r.status === 'initialized' || r.status === 'running') ||
            summBatches.some(b => b.status === 'running') ||
            retrievalBatches.some(b => b.status === 'running') ||
            rerankBatches.some(b => b.status === 'running');

        if (!hasRunning) return;

        const interval = setInterval(() => {
            if (allRuns.some(r => r.status === 'initialized' || r.status === 'running')) loadRuns();
            if (summBatches.some(b => b.status === 'running')) loadSummBatches();
            if (retrievalBatches.some(b => b.status === 'running')) loadRetrievalBatches();
            if (rerankBatches.some(b => b.status === 'running')) loadRerankBatches();
        }, 3000);

        return () => clearInterval(interval);
    }, [allRuns, summBatches, retrievalBatches, rerankBatches, loadRuns, loadSummBatches, loadRetrievalBatches, loadRerankBatches]);

    const openNewVersionModal = async (step: "summ" | "retrieval" | "rerank") => {
        setNewVersionName('');
        setSelectedRunId(undefined);
        setSelectedSummBatch(undefined);
        setSelectedRetrievalBatch(undefined);
        if (step === 'summ') await loadRuns();
        if (step === 'retrieval') await loadSummBatches();
        if (step === 'rerank') await loadRetrievalBatches();
        setShowNewVersionModal(step);
    };

    const handleViewRun = async (run: EvaluationRunResponse) => {
        setSelectedRun(run);
        setRunDetailLoading(true);
        setShowRunDetailModal(true);
        try {
            setRunConversations(await getRunConversations(run.id));
        } catch (e) {
            console.error('Failed to load conversations:', e);
            setRunConversations([]);
        } finally {
            setRunDetailLoading(false);
        }
    };

    const handleInit = async () => {
        setLoading(true);
        try {
            await initializeRun(dataset, samplePercent, newVersionName);
            setShowRunModal(false);
            setNewVersionName('');
            loadRuns();
        } catch (error) {
            console.error('Init failed:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleCreateProfiler = async (runIdOverride?: number | unknown) => {
        const runId = typeof runIdOverride === 'number' ? runIdOverride : selectedRunId;
        if (!runId) return;
        setLoading(true);
        try {
            await runProfilerStep(runId, newVersionName);
            setShowNewVersionModal(null);
            setNewVersionName('');
            loadSummBatches();
        } catch (error) {
            console.error('Profiler step failed:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleRunRetrieval = async (summBatchOverride?: number) => {
        const summBatchId = summBatchOverride ?? selectedSummBatch;
        const runId = summBatches.find(b => b.id === summBatchId)?.run_id ?? selectedRunId;
        if (!runId || !summBatchId) return;
        setLoading(true);
        try {
            await runRetrievalStep(runId, nSample, summBatchId, newVersionName);
            setShowNewVersionModal(null);
            setNewVersionName('');
            loadRetrievalBatches();
        } catch (error) {
            console.error('Retrieval step failed:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleRunReranking = async () => {
        const retriBatchId = selectedRetrievalBatch;
        const runId = retrievalBatches.find(b => b.id === retriBatchId)?.run_id ?? selectedRunId;
        if (!runId || !retriBatchId) return;
        setLoading(true);
        try {
            await runRerankingStep(runId, topK, model, retriBatchId, newVersionName);
            setShowNewVersionModal(null);
            setNewVersionName('');
            loadRerankBatches();
        } catch (error) {
            console.error('Reranking step failed:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleRunNextFromDetail = async (runId: number, batchId: number, stepType: string) => {
        if (stepType === 'summarization') {
            setShowDetailModal(false);
            setBatchDetail(null);
            setLoading(true);
            try {
                await runRetrievalStep(runId, nSample, batchId, undefined);
                loadRetrievalBatches();
            } catch (error) {
                console.error('Retrieval step failed:', error);
            } finally {
                setLoading(false);
            }
        } else if (stepType === 'retrieval') {
            setShowDetailModal(false);
            setBatchDetail(null);
            setLoading(true);
            try {
                await runRerankingStep(runId, topK, model, batchId, undefined);
                loadRerankBatches();
            } catch (error) {
                console.error('Reranking step failed:', error);
            } finally {
                setLoading(false);
            }
        }
    };

    const handleViewDetail = async (batch: BatchStepExecutionResponse) => {
        setDetailLoading(true);
        try {
            const detail = await getBatchDetail(batch.id, batch.step_type);
            setBatchDetail(detail);
            setShowDetailModal(true);
        } catch (error) {
            console.error('Failed to load detail:', error);
        } finally {
            setDetailLoading(false);
        }
    };

    const getRunIdFromSummBatch = (batchId: number | undefined) => {
        if (!batchId) return undefined;
        return summBatches.find(b => b.id === batchId)?.run_id;
    };

    // Count badges for tabs
    const counts: Record<TabId, number> = {
        init: stepRuns.length,
        profiler: summBatches.length,
        orchestrator: retrievalBatches.length,
        reranker: rerankBatches.length,
    };

    const activeTabData = TABS.find(t => t.id === activeTab)!;

    return (
        <div className="min-h-screen bg-gray-50/50">
            <div className="container mx-auto px-4 py-8 max-w-7xl">

                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900 mb-1">Step-by-Step Optimization</h1>
                    <p className="text-gray-500 text-sm">
                        Run each pipeline stage independently and compare results across versions.
                    </p>
                </div>

                {/* Main layout: left (tabs + content) + right (pipeline diagram) */}
                <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-6 items-start">

                    {/* ── Left column ── */}
                    <div>
                        {/* Tab bar */}
                        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-1.5 flex gap-1 mb-4">
                            {TABS.map(tab => {
                                const isActive = activeTab === tab.id;
                                return (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        className={`flex-1 flex flex-col items-center gap-1 px-2 py-2.5 rounded-xl text-center transition-all ${isActive ? `${tab.color.bg} ${tab.color.ring} border-2 shadow-sm` : 'border-2 border-transparent hover:bg-gray-50'}`}
                                    >
                                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-lg ${isActive ? tab.color.badge : 'bg-gray-100'}`}>
                                            {tab.icon}
                                        </div>
                                        <div>
                                            <div className={`text-[10px] font-bold uppercase tracking-wider ${isActive ? tab.color.text : 'text-gray-400'}`}>{tab.step}</div>
                                            <div className={`text-xs font-semibold leading-tight ${isActive ? tab.color.text : 'text-gray-600'}`}>{tab.title}</div>
                                        </div>
                                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${isActive ? `${tab.color.badge} ${tab.color.text}` : 'bg-gray-100 text-gray-500'}`}>
                                            {counts[tab.id]}
                                        </span>
                                    </button>
                                );
                            })}
                        </div>

                        {/* Tab content */}
                        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
                            {/* Content header */}
                            <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-white">
                                <div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs font-bold uppercase tracking-wider text-gray-400">{activeTabData.step}</span>
                                        <h2 className="text-base font-bold text-gray-800">{activeTabData.title}</h2>
                                    </div>
                                    <p className="text-xs text-gray-500 mt-0.5">{activeTabData.subtitle}</p>
                                </div>

                                {/* Action button per tab */}
                                {activeTab === 'init' && (
                                    <button
                                        onClick={() => { setNewVersionName(''); setShowRunModal(true); }}
                                        className="flex items-center gap-2 bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-xl transition-colors text-sm font-medium shadow-sm"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
                                        New Run
                                    </button>
                                )}
                                {activeTab === 'profiler' && (
                                    <button
                                        onClick={() => openNewVersionModal('summ')}
                                        className="flex items-center gap-2 bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-xl transition-colors text-sm font-medium shadow-sm"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
                                        New Version
                                    </button>
                                )}
                                {activeTab === 'orchestrator' && (
                                    <button
                                        onClick={() => openNewVersionModal('retrieval')}
                                        className="flex items-center gap-2 bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-xl transition-colors text-sm font-medium shadow-sm"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
                                        New Version
                                    </button>
                                )}
                                {activeTab === 'reranker' && (
                                    <button
                                        onClick={() => openNewVersionModal('rerank')}
                                        className="flex items-center gap-2 bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-xl transition-colors text-sm font-medium shadow-sm"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
                                        New Version
                                    </button>
                                )}
                            </div>

                            {/* Content body */}
                            <div className="px-6 py-5">

                                {/* Step 1: Evaluation Runs */}
                                {activeTab === 'init' && (
                                    <div className="space-y-3">
                                        {stepRuns.length === 0 && <EmptyState icon="🚀" message="No runs yet. Click '+ New Run' to create one." />}
                                        {stepRuns.map(run => (
                                            <div
                                                key={run.id}
                                                onClick={() => handleViewRun(run)}
                                                className="p-4 bg-white border border-gray-200 rounded-2xl hover:border-gray-300 hover:shadow-sm transition-all cursor-pointer group"
                                            >
                                                <div className="flex justify-between items-start mb-2">
                                                    <div className="flex items-center gap-2">
                                                        <span className="font-semibold text-gray-800">
                                                            {run.name ? run.name : `Run #${run.id}`}
                                                        </span>
                                                        {run.name && <span className="text-xs text-gray-400">#{run.id}</span>}
                                                        <span className={`text-xs px-2 py-0.5 rounded-full capitalize font-medium ${run.status === 'completed' ? 'bg-green-100 text-green-700' : run.status === 'failed' ? 'bg-red-100 text-red-600' : 'bg-amber-100 text-amber-700'}`}>
                                                            {run.status}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-xs text-gray-400">{run.timestamp ? new Date(run.timestamp).toLocaleString() : ''}</span>
                                                        <span className="text-xs text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity">View →</span>
                                                    </div>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <span className="bg-white px-2 py-0.5 rounded-lg border border-gray-200 text-xs">Dataset: <strong className="capitalize">{run.dataset}</strong></span>
                                                    <span className="bg-white px-2 py-0.5 rounded-lg border border-gray-200 text-xs">Samples: <strong>{run.sample_size}</strong></span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* Step 2: Profiler Agent */}
                                {activeTab === 'profiler' && (
                                    <div className="space-y-3">
                                        {summBatches.length === 0 && <EmptyState icon="🤖" message="No Profiler Agent versions yet. Click '+ New Version' to create one." />}
                                        {summBatches.map(b => (
                                            <VersionCard key={b.id} batch={b} onClick={() => handleViewDetail(b)} />
                                        ))}
                                    </div>
                                )}

                                {/* Step 3: Orchestrator Agent */}
                                {activeTab === 'orchestrator' && (
                                    <div className="space-y-3">
                                        {retrievalBatches.length === 0 && <EmptyState icon="🎛️" message="No Orchestrator Agent versions yet. Click '+ New Version' to create one." />}
                                        {retrievalBatches.map(b => (
                                            <VersionCard key={b.id} batch={b} onClick={() => handleViewDetail(b)}>
                                                <div className="flex gap-3 text-xs text-gray-600">
                                                    <span className="bg-white px-2 py-0.5 rounded border border-gray-200">N = <strong>{(b.config as Record<string, unknown>).n_sample as number}</strong> candidates</span>
                                                </div>
                                            </VersionCard>
                                        ))}
                                    </div>
                                )}

                                {/* Step 4: Reranker */}
                                {activeTab === 'reranker' && (
                                    <div className="space-y-3">
                                        {rerankBatches.length === 0 && <EmptyState icon="🎯" message="No Reranker versions yet. Click '+ New Version' to create one." />}
                                        {rerankBatches.map(b => (
                                            <VersionCard key={b.id} batch={b} onClick={() => handleViewDetail(b)}>
                                                <div className="flex gap-2 text-xs text-gray-600">
                                                    <span className="bg-white px-2 py-0.5 rounded border border-gray-200">Top-K = <strong>{(b.config as Record<string, unknown>).top_k as number}</strong></span>
                                                    <span className="bg-white px-2 py-0.5 rounded border border-gray-200 capitalize">{String((b.config as Record<string, unknown>).model)}</span>
                                                </div>
                                            </VersionCard>
                                        ))}
                                    </div>
                                )}

                            </div>
                        </div>
                    </div>

                    {/* ── Right column: Pipeline diagram (sticky) ── */}
                    <div className="sticky top-6">
                        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm px-4 py-4">
                            <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-4">ARGOS Pipeline</p>

                            <VerticalPipelineStep
                                step="Step 1" icon="🚀" title="Initialize" subtitle="Dataset sample"
                                color={TABS[0].color}
                                active={activeTab === 'init'}
                                onClick={() => setActiveTab('init')}
                            />
                            <PipelineConnector active={activeTab === 'profiler' || activeTab === 'orchestrator' || activeTab === 'reranker'} />

                            <VerticalPipelineStep
                                step="Step 2" icon="🤖" title="Profiler Agent" subtitle="Extract preferences"
                                color={TABS[1].color}
                                active={activeTab === 'profiler'}
                                onClick={() => setActiveTab('profiler')}
                            />
                            <PipelineConnector active={activeTab === 'orchestrator' || activeTab === 'reranker'} />

                            <VerticalPipelineStep
                                step="Step 3" icon="🎛️" title="Orchestrator Agent" subtitle="Retrieve & filter"
                                color={TABS[2].color}
                                active={activeTab === 'orchestrator'}
                                onClick={() => setActiveTab('orchestrator')}
                            />
                            {activeTab === 'orchestrator' && <OrchestratorDetail />}
                            <PipelineConnector active={activeTab === 'reranker'} />

                            <VerticalPipelineStep
                                step="Step 4" icon="🎯" title="Reranker" subtitle="Recall@K"
                                color={TABS[3].color}
                                active={activeTab === 'reranker'}
                                onClick={() => setActiveTab('reranker')}
                            />
                        </div>

                    </div>

                </div>
            </div>

            {/* === MODALS === */}

            {/* Loading overlay */}
            {(detailLoading || loading) && (
                <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/30 backdrop-blur-sm">
                    <div className="bg-white rounded-2xl px-8 py-5 shadow-xl flex items-center gap-4">
                        <svg className="animate-spin h-5 w-5 text-gray-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        <span className="text-gray-700 font-medium">{detailLoading ? 'Loading details...' : 'Running step...'}</span>
                    </div>
                </div>
            )}

            {/* Version Detail Modal */}
            {showDetailModal && batchDetail && (
                <VersionDetailModal
                    detail={batchDetail}
                    onClose={() => { setShowDetailModal(false); setBatchDetail(null); }}
                    onRunNextStep={batchDetail.step_type !== 'reranking' ? (runId, batchId) => {
                        handleRunNextFromDetail(runId, batchId, batchDetail.step_type);
                    } : undefined}
                    nextStepConfig={
                        batchDetail.step_type === 'summarization'
                            ? { nSample, setNSample }
                            : batchDetail.step_type === 'retrieval'
                                ? { topK, setTopK, model, setModel }
                                : undefined
                    }
                    loading={loading}
                />
            )}

            {/* Run Detail Modal */}
            {showRunDetailModal && selectedRun && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => { setShowRunDetailModal(false); setRunConversations([]); }}>
                    <div className="bg-white rounded-2xl shadow-2xl w-[92vw] max-w-3xl flex flex-col max-h-[85vh]" onClick={e => e.stopPropagation()}>
                        <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center shrink-0">
                            <div>
                                <h3 className="text-lg font-semibold text-gray-800">
                                    {selectedRun.name ? selectedRun.name : `Run #${selectedRun.id}`}
                                    {selectedRun.name && <span className="ml-2 text-sm font-normal text-gray-400">Run #{selectedRun.id}</span>}
                                </h3>
                                <p className="text-sm text-gray-500 mt-0.5">
                                    {selectedRun.dataset} · {selectedRun.sample_size} conversations
                                </p>
                            </div>
                            <button onClick={() => { setShowRunDetailModal(false); setRunConversations([]); }} className="text-gray-400 hover:text-gray-600 text-2xl leading-none w-8 h-8 flex items-center justify-center rounded-lg hover:bg-gray-100 transition-colors">&times;</button>
                        </div>
                        <div className="overflow-y-auto flex-1 px-6 py-4">
                            {runDetailLoading ? (
                                <div className="flex items-center justify-center py-16 gap-3">
                                    <svg className="animate-spin h-5 w-5 text-gray-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                    </svg>
                                    <span className="text-gray-500">Loading conversations...</span>
                                </div>
                            ) : runConversations.length === 0 ? (
                                <div className="text-center py-16 text-gray-400">No conversations found for this run.</div>
                            ) : (
                                <div className="space-y-3">
                                    {runConversations.map((conv, idx) => (
                                        <div key={conv.id} className="p-3 bg-gray-50 border border-gray-200 rounded-xl">
                                            <div className="flex justify-between items-center mb-1">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-xs font-mono text-gray-400">#{idx + 1}</span>
                                                    <span className="text-sm font-medium text-gray-800 truncate max-w-[200px]">{conv.conv_id}</span>
                                                    <span className={`text-xs px-1.5 py-0.5 rounded-full capitalize ${conv.status === 'completed' ? 'bg-green-100 text-green-700' : conv.status === 'pending' ? 'bg-yellow-100 text-yellow-700' : 'bg-gray-100 text-gray-600'}`}>
                                                        {conv.status}
                                                    </span>
                                                </div>
                                                <span className="text-xs text-gray-500 shrink-0">
                                                    Target: <strong>{Array.isArray(conv.target) ? conv.target.join(', ') : conv.target}</strong>
                                                </span>
                                            </div>
                                            {conv.liked_movies && conv.liked_movies.length > 0 && (
                                                <p className="text-xs text-gray-600 mb-1">❤️ {conv.liked_movies.join(', ')}</p>
                                            )}
                                            {conv.dialog_preview && (
                                                <p className="text-xs text-gray-500 line-clamp-2">{conv.dialog_preview}…</p>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                        <div className="px-6 py-3 border-t border-gray-100 flex justify-end shrink-0 bg-gray-50/50 rounded-b-2xl">
                            <button
                                onClick={() => { setShowRunDetailModal(false); setRunConversations([]); }}
                                className="px-4 py-2 text-sm bg-white border border-gray-200 hover:bg-gray-50 rounded-lg transition-colors text-gray-600"
                            >Close</button>
                        </div>
                    </div>
                </div>
            )}

            {/* Init Run Modal */}
            {showRunModal && (
                <ModalBackdrop
                    title="Initialize New Run"
                    onClose={() => setShowRunModal(false)}
                    onSubmit={handleInit}
                    submitLabel={loading ? 'Initializing...' : 'Initialize'}
                    submitDisabled={loading}
                >
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Version Name <span className="text-gray-400">(optional)</span></label>
                            <input
                                type="text"
                                value={newVersionName}
                                onChange={(e) => setNewVersionName(e.target.value)}
                                placeholder="e.g. initial-run..."
                                className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Dataset</label>
                            <select
                                value={dataset}
                                onChange={(e) => setDataset(e.target.value as "redial" | "inspired")}
                                className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                            >
                                <option value="redial">Redial</option>
                                <option value="inspired">Inspired</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Sample Size
                                <span className="ml-2 font-bold text-gray-900">{samplePercent}%</span>
                                {datasetSizes[dataset] ? (
                                    <span className="ml-1 text-gray-400 font-normal">
                                        (~{Math.max(1, Math.round(datasetSizes[dataset] * samplePercent / 100))} conversations)
                                    </span>
                                ) : null}
                            </label>
                            <input
                                type="range"
                                min={1}
                                max={100}
                                value={samplePercent}
                                onChange={(e) => setSamplePercent(Number(e.target.value))}
                                className="w-full accent-gray-800"
                            />
                            <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                                <span>1%</span><span>50%</span><span>100%</span>
                            </div>
                        </div>
                    </div>
                </ModalBackdrop>
            )}

            {/* New Profiler Agent Version Modal */}
            {showNewVersionModal === 'summ' && (
                <ModalBackdrop
                    title="New Profiler Agent Version"
                    onClose={() => setShowNewVersionModal(null)}
                    onSubmit={() => handleCreateProfiler()}
                    submitLabel={loading ? 'Running...' : 'Run Profiler Agent'}
                    submitDisabled={loading || !selectedRunId}
                >
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Version Name <span className="text-gray-400">(optional)</span></label>
                        <input
                            type="text"
                            value={newVersionName}
                            onChange={(e) => setNewVersionName(e.target.value)}
                            placeholder="e.g. baseline, experiment-1..."
                            className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Select Run <span className="text-red-500">*</span></label>
                        <select
                            value={selectedRunId || ''}
                            onChange={(e) => setSelectedRunId(e.target.value ? Number(e.target.value) : undefined)}
                            className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                        >
                            <option value="">— Select a run —</option>
                            {stepRuns.map(r => (
                                <option key={r.id} value={r.id}>
                                    {r.name ? r.name : `Run #${r.id}`} — {r.dataset} ({r.sample_size} samples) [{r.status}]
                                </option>
                            ))}
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Profiler Agent will extract user preferences from all conversations in this run.</p>
                    </div>
                    <div className="bg-gray-50 border border-gray-200 rounded-xl p-3 text-sm text-gray-700">
                        <p className="font-medium mb-1">ℹ️ No additional parameters needed</p>
                        <p className="text-gray-500 text-xs">Extracts preferences, genres, hard constraints, and WRRF weights from conversation history.</p>
                    </div>
                </ModalBackdrop>
            )}

            {/* New Orchestrator Agent Version Modal */}
            {showNewVersionModal === 'retrieval' && (
                <ModalBackdrop
                    title="New Orchestrator Agent Version"
                    onClose={() => setShowNewVersionModal(null)}
                    onSubmit={() => {
                        const runId = getRunIdFromSummBatch(selectedSummBatch) || selectedRunId;
                        if (!runId) return;
                        setSelectedRunId(runId);
                        handleRunRetrieval(selectedSummBatch);
                    }}
                    submitLabel={loading ? 'Running...' : 'Run Orchestrator Agent'}
                    submitDisabled={loading || !selectedSummBatch}
                >
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Version Name <span className="text-gray-400">(optional)</span></label>
                        <input
                            type="text"
                            value={newVersionName}
                            onChange={(e) => setNewVersionName(e.target.value)}
                            placeholder="e.g. n100-critic-v1..."
                            className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Input Profiler Agent Version <span className="text-red-500">*</span></label>
                        <select
                            value={selectedSummBatch || ''}
                            onChange={(e) => {
                                const batchId = e.target.value ? Number(e.target.value) : undefined;
                                setSelectedSummBatch(batchId);
                                if (batchId) {
                                    const batch = summBatches.find(b => b.id === batchId);
                                    if (batch) setSelectedRunId(batch.run_id);
                                }
                            }}
                            className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                        >
                            <option value="">— Select a profiler version —</option>
                            {summBatches.map(b => (
                                <option key={b.id} value={b.id}>
                                    {b.name ? b.name : `v${b.version}`} (Run #{b.run_id}) — {b.status} — {new Date(b.created_at).toLocaleString()}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">N Candidates</label>
                        <input
                            type="number"
                            value={nSample}
                            onChange={(e) => setNSample(Number(e.target.value))}
                            className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                        />
                        <p className="text-xs text-gray-500 mt-1">Number of candidates to retrieve (10–600).</p>
                    </div>
                    <div className="bg-gray-50 border border-gray-200 rounded-xl p-3 text-xs text-gray-600">
                        Orchestrator kích hoạt 3 luồng truy xuất song song (Semantic / Content / Graph), hợp nhất qua WRRF với trọng số động, rồi Critic lọc ràng buộc cứng — Relaxation Agent can thiệp nếu cần.
                    </div>
                </ModalBackdrop>
            )}

            {/* New Reranker Version Modal */}
            {showNewVersionModal === 'rerank' && (
                <ModalBackdrop
                    title="New Reranker Version"
                    onClose={() => setShowNewVersionModal(null)}
                    onSubmit={handleRunReranking}
                    submitLabel={loading ? 'Running...' : 'Run Reranker'}
                    submitDisabled={loading || !selectedRetrievalBatch}
                >
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Version Name <span className="text-gray-400">(optional)</span></label>
                        <input
                            type="text"
                            value={newVersionName}
                            onChange={(e) => setNewVersionName(e.target.value)}
                            placeholder="e.g. cohere-top10..."
                            className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Input Orchestrator Agent Version <span className="text-red-500">*</span></label>
                        <select
                            value={selectedRetrievalBatch || ''}
                            onChange={(e) => {
                                const batchId = e.target.value ? Number(e.target.value) : undefined;
                                setSelectedRetrievalBatch(batchId);
                                if (batchId) {
                                    const batch = retrievalBatches.find(b => b.id === batchId);
                                    if (batch) setSelectedRunId(batch.run_id);
                                }
                            }}
                            className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                        >
                            <option value="">— Select an orchestrator version —</option>
                            {retrievalBatches.filter(b => b.status === 'completed').map(b => (
                                <option key={b.id} value={b.id}>
                                    {b.name ? b.name : `v${b.version}`} (Run #{b.run_id}) — {new Date(b.created_at).toLocaleString()}
                                </option>
                            ))}
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Only completed Orchestrator Agent versions are shown.</p>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Top K</label>
                            <input
                                type="number"
                                value={topK}
                                onChange={(e) => setTopK(Number(e.target.value))}
                                className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
                            <select
                                value={model}
                                onChange={(e) => setModel(e.target.value as "llm" | "cohere")}
                                className="w-full rounded-xl border border-gray-200 focus:border-gray-400 outline-none px-3 py-2 text-sm bg-white transition-colors"
                            >
                                <option value="cohere">Cohere</option>
                                <option value="llm">LLM</option>
                            </select>
                        </div>
                    </div>
                    <div className="bg-gray-50 border border-gray-200 rounded-xl p-3 text-xs text-gray-600">
                        Sử dụng danh sách ứng viên đã lọc từ Orchestrator Agent (Step 3) để xếp hạng lại theo mức độ phù hợp với người dùng.
                    </div>
                </ModalBackdrop>
            )}
        </div>
    );
}
