import { useState, useEffect, useCallback } from 'react';
import { initializeRun, runProfilerStep, runRetrievalStep, runRerankingStep, getStepsByType, getBatchDetail, getEvaluationRuns, getRunConversations, BatchStepExecutionResponse, BatchDetailResponse, EvaluationRunResponse, ConversationLogItem } from '@/lib/api';
import VersionDetailModal from '@/components/evaluation/VersionDetailModal';

// ─── Shared UI components (defined OUTSIDE the page to keep stable references) ───

const TabButton = ({ id, label, activeTab, setActiveTab }: { id: string; label: string; activeTab: string; setActiveTab: (id: string) => void }) => (
    <button
        onClick={() => setActiveTab(id)}
        className={`px-4 py-2 font-medium text-sm rounded-t-lg transition-colors ${activeTab === id
            ? 'bg-white text-blue-600 border-t border-x border-gray-200'
            : 'bg-gray-50 text-gray-500 hover:text-gray-700'
            }`}
    >
        {label}
    </button>
);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const VersionCard = ({ batch, color, onClick, children }: { batch: BatchStepExecutionResponse; color: { bg: string; border: string; text: string }; onClick: () => void; children?: any }) => (
    <div
        onClick={onClick}
        className={`p-4 ${color.bg} border ${color.border} rounded-lg cursor-pointer hover:shadow-md transition-all`}
    >
        <div className="flex justify-between items-start mb-2">
            <div className="flex items-center gap-3">
                <span className={`text-base font-bold ${color.text}`}>
                    {batch.name ? batch.name : `v${batch.version}`}
                </span>
                {batch.name && <span className="text-xs text-gray-400">v{batch.version}</span>}
            </div>
            <span className="text-xs text-gray-500">{new Date(batch.created_at).toLocaleString()}</span>
        </div>
        <div className="flex justify-between items-center mb-2">
            <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">Run #{batch.run_id}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full capitalize ${batch.status === 'completed' ? 'bg-green-100 text-green-700' : batch.status === 'failed' ? 'bg-red-100 text-red-600' : 'bg-yellow-100 text-yellow-700'}`}>
                {batch.status}
            </span>
        </div>
        {children && <div className="mt-2">{children}</div>}
    </div>
);

const ModalBackdrop = ({ title, onClose, children, onSubmit, submitLabel, submitColor, submitDisabled }: {
    title: string; onClose: () => void; children: React.ReactNode;
    onSubmit: () => void; submitLabel: string; submitColor: string; submitDisabled?: boolean;
}) => (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
        <div className="bg-white rounded-xl shadow-2xl w-[90vw] max-w-lg flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
                <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl leading-none">&times;</button>
            </div>
            <div className="px-6 py-5 space-y-4">
                {children}
            </div>
            <div className="px-6 py-4 border-t border-gray-200 flex justify-end gap-3">
                <button onClick={onClose} className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors">Cancel</button>
                <button onClick={(e) => { e.preventDefault(); onSubmit(); }} disabled={submitDisabled} className={`px-6 py-2 text-sm text-white rounded-lg ${submitColor} disabled:opacity-50 transition-colors`}>
                    {submitLabel}
                </button>
            </div>
        </div>
    </div>
);

const EmptyState = ({ icon, message }: { icon: string; message: string }) => (
    <div className="text-center py-16">
        <div className="text-5xl mb-4">{icon}</div>
        <p className="text-gray-500">{message}</p>
    </div>
);

const NewVersionButton = ({ onClick, color }: { onClick: () => void; color: string }) => (
    <button
        onClick={onClick}
        className={`flex items-center gap-2 ${color} text-white px-4 py-2 rounded-lg transition-colors text-sm font-medium shadow-sm`}
    >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
        New Version
    </button>
);

export default function StepEvaluationPage() {
    const [loading, setLoading] = useState(false);
    const [activeTab, setActiveTab] = useState<"init" | "summ" | "retr" | "rerank">("init");

    // Init State (for modal)
    const [dataset, setDataset] = useState<"inspired" | "redial">("redial");
    const [sampleSize, setSampleSize] = useState(5);

    // Retrieval State (for modal)
    const [nSample, setNSample] = useState(100);

    // Rerank State (for modal)
    const [topK, setTopK] = useState(10);
    const [model, setModel] = useState<"llm" | "cohere">("cohere");

    // Data per tab
    const [allRuns, setAllRuns] = useState<EvaluationRunResponse[]>([]);
    const [summBatches, setSummBatches] = useState<BatchStepExecutionResponse[]>([]);
    const [retrBatches, setRetrBatches] = useState<BatchStepExecutionResponse[]>([]);
    const [rerankBatches, setRerankBatches] = useState<BatchStepExecutionResponse[]>([]);

    // Detail Modal State
    const [showDetailModal, setShowDetailModal] = useState(false);
    const [batchDetail, setBatchDetail] = useState<BatchDetailResponse | null>(null);
    const [detailLoading, setDetailLoading] = useState(false);

    // New Version Modal State
    const [showNewVersionModal, setShowNewVersionModal] = useState<"summ" | "retr" | "rerank" | null>(null);
    const [newVersionName, setNewVersionName] = useState('');
    const [selectedRunId, setSelectedRunId] = useState<number | undefined>(undefined);
    const [selectedSummBatch, setSelectedSummBatch] = useState<number | undefined>(undefined);
    const [selectedRetrBatch, setSelectedRetrBatch] = useState<number | undefined>(undefined);

    // Run Modal State (for Init only)
    const [showRunModal, setShowRunModal] = useState(false);

    // Run Detail Modal State (conversation list)
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

    const loadRetrBatches = useCallback(async () => {
        try { setRetrBatches(await getStepsByType('retrieval')); } catch (e) { console.error(e); }
    }, []);

    const loadRerankBatches = useCallback(async () => {
        try { setRerankBatches(await getStepsByType('reranking')); } catch (e) { console.error(e); }
    }, []);

    // Load tab data when switching
    useEffect(() => {
        if (activeTab === 'init') loadRuns();
        if (activeTab === 'summ') loadSummBatches();
        if (activeTab === 'retr') loadRetrBatches();
        if (activeTab === 'rerank') loadRerankBatches();
    }, [activeTab, loadRuns, loadSummBatches, loadRetrBatches, loadRerankBatches]);

    // --- Polling Effect ---
    useEffect(() => {
        let interval: NodeJS.Timeout;

        const checkRunning = () => {
            if (activeTab === 'init') return allRuns.some(r => r.status === 'initialized' || r.status === 'running');
            if (activeTab === 'summ') return summBatches.some(b => b.status === 'running');
            if (activeTab === 'retr') return retrBatches.some(b => b.status === 'running');
            if (activeTab === 'rerank') return rerankBatches.some(b => b.status === 'running');
            return false;
        };

        if (checkRunning()) {
            interval = setInterval(() => {
                if (activeTab === 'init') loadRuns();
                if (activeTab === 'summ') loadSummBatches();
                if (activeTab === 'retr') loadRetrBatches();
                if (activeTab === 'rerank') loadRerankBatches();
            }, 3000);
        }

        return () => {
            if (interval) clearInterval(interval);
        };
    }, [activeTab, allRuns, summBatches, retrBatches, rerankBatches, loadRuns, loadSummBatches, loadRetrBatches, loadRerankBatches]);


    // --- Open New Version Modal (preload prev step data) ---
    const openNewVersionModal = async (step: "summ" | "retr" | "rerank") => {
        setNewVersionName('');
        setSelectedRunId(undefined);
        setSelectedSummBatch(undefined);
        setSelectedRetrBatch(undefined);
        // Preload previous step data for the dropdown
        if (step === 'summ') await loadRuns();
        if (step === 'retr') await loadSummBatches();
        if (step === 'rerank') await loadRetrBatches();
        setShowNewVersionModal(step);
    };

    // --- Handlers ---
    const handleViewRun = async (run: EvaluationRunResponse) => {
        setSelectedRun(run);
        setRunDetailLoading(true);
        setShowRunDetailModal(true);
        try {
            const convs = await getRunConversations(run.id);
            setRunConversations(convs);
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
            await initializeRun(dataset, sampleSize, 0, newVersionName);
            setShowRunModal(false);
            setNewVersionName('');
            loadRuns();
        } catch (error) {
            console.error('Init failed:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleCreateProfiler = async (runIdOverride?: number | any) => {
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

    const handleCreateRetrieval = async (runIdOverride?: number | any) => {
        const runId = typeof runIdOverride === 'number' ? runIdOverride : selectedRunId;
        if (!runId) return;
        setLoading(true);
        try {
            await runRetrievalStep(runId, nSample, selectedSummBatch, newVersionName);
            setShowNewVersionModal(null);
            setNewVersionName('');
            loadRetrBatches();
        } catch (error) {
            console.error('Retrieval failed:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleCreateReranking = async (runIdOverride?: number | any) => {
        const runId = typeof runIdOverride === 'number' ? runIdOverride : selectedRunId;
        if (!runId) return;
        setLoading(true);
        try {
            await runRerankingStep(runId, topK, model, selectedRetrBatch, newVersionName);
            setShowNewVersionModal(null);
            setNewVersionName('');
            loadRerankBatches();
        } catch (error) {
            console.error('Reranking failed:', error);
        } finally {
            setLoading(false);
        }
    };

    // Handle from detail modal "Run Next Step"
    const handleRunNextFromDetail = async (runId: number, batchId: number, stepType: string) => {
        setLoading(true);
        try {
            if (stepType === 'summarization') {
                await runRetrievalStep(runId, nSample, batchId);
                setShowDetailModal(false);
                setBatchDetail(null);
                setActiveTab('retr');
                loadRetrBatches();
            } else if (stepType === 'retrieval') {
                await runRerankingStep(runId, topK, model, batchId);
                setShowDetailModal(false);
                setBatchDetail(null);
                setActiveTab('rerank');
                loadRerankBatches();
            }
        } catch (error) {
            console.error('Next step failed:', error);
        } finally {
            setLoading(false);
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

    // --- UI Components ---
    // (TabButton, VersionCard, ModalBackdrop, EmptyState, NewVersionButton are defined at module level)
    const getRunIdFromSummBatch = (batchId: number | undefined) => {
        if (!batchId) return undefined;
        const batch = summBatches.find(b => b.id === batchId);
        return batch?.run_id;
    };
    const getRunIdFromRetrBatch = (batchId: number | undefined) => {
        if (!batchId) return undefined;
        const batch = retrBatches.find(b => b.id === batchId);
        return batch?.run_id;
    };

    return (
        <div className="container mx-auto px-4 py-8 max-w-4xl">
            <h1 className="text-3xl font-bold mb-6">Step-by-Step Optimization</h1>

            {/* Tabs */}
            <div className="flex border-b border-gray-200 mb-6">
                <TabButton id="init" label="1. Initialize" activeTab={activeTab} setActiveTab={(id) => setActiveTab(id as typeof activeTab)} />
                <TabButton id="summ" label="2. Profiler Agent" activeTab={activeTab} setActiveTab={(id) => setActiveTab(id as typeof activeTab)} />
                <TabButton id="retr" label="3. Retrieval" activeTab={activeTab} setActiveTab={(id) => setActiveTab(id as typeof activeTab)} />
                <TabButton id="rerank" label="4. Critic + Reranker" activeTab={activeTab} setActiveTab={(id) => setActiveTab(id as typeof activeTab)} />
            </div>

            {/* Tab Content */}
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 min-h-[400px]">

                {/* 1. Init Tab — List of Runs */}
                {activeTab === 'init' && (
                    <div>
                        <div className="flex justify-between items-center mb-6">
                            <div>
                                <h2 className="text-xl font-semibold text-gray-800">Evaluation Runs</h2>
                                <p className="text-sm text-gray-500 mt-1">Create runs with dataset samples.</p>
                            </div>
                            <button
                                onClick={() => setShowRunModal(true)}
                                className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium shadow-sm"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
                                New Run
                            </button>
                        </div>
                        <div className="space-y-3">
                            {allRuns.length === 0 && <EmptyState icon="🚀" message="No runs yet. Click '+ New Run' to create one." />}
                            {allRuns.map(run => (
                                <div
                                    key={run.id}
                                    onClick={() => handleViewRun(run)}
                                    className="p-4 bg-blue-50 border border-blue-100 rounded-lg hover:shadow-md hover:border-blue-300 transition-all cursor-pointer"
                                >
                                    <div className="flex justify-between items-start mb-2">
                                        <div className="flex items-center gap-3">
                                            <span className="font-semibold text-blue-900">
                                                {run.name ? run.name : `Run #${run.id}`}
                                            </span>
                                            {run.name && <span className="text-xs text-gray-400">Run #{run.id}</span>}
                                            <span className={`text-xs px-2 py-0.5 rounded-full capitalize ${run.status === 'completed' ? 'bg-green-100 text-green-700' : run.status === 'failed' ? 'bg-red-100 text-red-600' : 'bg-yellow-100 text-yellow-700'}`}>
                                                {run.status}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-xs text-gray-500">{run.timestamp ? new Date(run.timestamp).toLocaleString() : ''}</span>
                                            <span className="text-xs text-blue-400">👁 View</span>
                                        </div>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="bg-white px-2 py-0.5 rounded border border-gray-200">Dataset: <strong className="capitalize">{run.dataset}</strong></span>
                                        <span className="bg-white px-2 py-0.5 rounded border border-gray-200">Samples: <strong>{run.sample_size}</strong></span>
                                        {run.avg_recall != null && run.avg_recall > 0 && (
                                            <span className="bg-green-50 px-2 py-0.5 rounded border border-green-200">Recall: <strong>{run.avg_recall.toFixed(4)}</strong></span>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* 2. Profiler Agent Tab */}
                {activeTab === 'summ' && (
                    <div>
                        <div className="flex justify-between items-center mb-6">
                            <div>
                                <h2 className="text-xl font-semibold text-gray-800">Profiler Agent Versions</h2>
                                <p className="text-sm text-gray-500 mt-1">Click a version to view extracted preferences and create retrieval.</p>
                            </div>
                            <NewVersionButton onClick={() => openNewVersionModal('summ')} color="bg-indigo-600 hover:bg-indigo-700" />
                        </div>
                        <div className="space-y-3">
                            {summBatches.length === 0 && <EmptyState icon="🤖" message="No Profiler Agent versions yet. Click '+ New Version' to create one." />}
                            {summBatches.map(b => (
                                <VersionCard key={b.id} batch={b} color={{ bg: 'bg-indigo-50', border: 'border-indigo-100', text: 'text-indigo-800' }} onClick={() => handleViewDetail(b)} />
                            ))}
                        </div>
                    </div>
                )}

                {/* 3. Retrieval Tab */}
                {activeTab === 'retr' && (
                    <div>
                        <div className="flex justify-between items-center mb-6">
                            <div>
                                <h2 className="text-xl font-semibold text-gray-800">Retrieval Versions</h2>
                                <p className="text-sm text-gray-500 mt-1">Click a version to view details and create reranking.</p>
                            </div>
                            <NewVersionButton onClick={() => openNewVersionModal('retr')} color="bg-purple-600 hover:bg-purple-700" />
                        </div>
                        <div className="space-y-3">
                            {retrBatches.length === 0 && <EmptyState icon="🔍" message="No retrieval versions yet. Click '+ New Version' to create one." />}
                            {retrBatches.map(b => (
                                <VersionCard key={b.id} batch={b} color={{ bg: 'bg-purple-50', border: 'border-purple-100', text: 'text-purple-800' }} onClick={() => handleViewDetail(b)}>
                                    <div className="flex gap-3 text-xs text-gray-600">
                                        <span>N={(b.config as Record<string, unknown>).n_sample as number}</span>
                                        {!!(b.config as Record<string, unknown>).input_batch && <span className="text-gray-400">(Input: Batch #{String((b.config as Record<string, unknown>).input_batch)})</span>}
                                    </div>
                                </VersionCard>
                            ))}
                        </div>
                    </div>
                )}

                {/* 4. Critic + Reranker Tab */}
                {activeTab === 'rerank' && (
                    <div>
                        <div className="flex justify-between items-center mb-6">
                            <div>
                                <h2 className="text-xl font-semibold text-gray-800">Critic + Reranker Versions</h2>
                                <p className="text-sm text-gray-500 mt-1">Critic filters candidates, then Reranker selects top-K. Click a version to view recall results.</p>
                            </div>
                            <NewVersionButton onClick={() => openNewVersionModal('rerank')} color="bg-green-600 hover:bg-green-700" />
                        </div>
                        <div className="space-y-3">
                            {rerankBatches.length === 0 && <EmptyState icon="🏆" message="No Critic + Reranker versions yet. Click '+ New Version' to create one." />}
                            {rerankBatches.map(b => (
                                <VersionCard key={b.id} batch={b} color={{ bg: 'bg-green-50', border: 'border-green-100', text: 'text-green-800' }} onClick={() => handleViewDetail(b)}>
                                    <div className="flex gap-3 text-xs text-gray-600">
                                        <span>TopK={(b.config as Record<string, unknown>).top_k as number}</span>
                                        <span>{String((b.config as Record<string, unknown>).model)}</span>
                                        {!!(b.config as Record<string, unknown>).input_batch && <span className="text-gray-400">(Input: Batch #{String((b.config as Record<string, unknown>).input_batch)})</span>}
                                    </div>
                                </VersionCard>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* === MODALS === */}

            {/* Loading overlay */}
            {(detailLoading || loading) && (
                <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/30">
                    <div className="bg-white rounded-lg px-6 py-4 shadow-lg flex items-center gap-3">
                        <svg className="animate-spin h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        <span className="text-gray-700">{detailLoading ? 'Loading details...' : 'Running step...'}</span>
                    </div>
                </div>
            )}

            {/* Version Detail Modal (with next-step action) */}
            {showDetailModal && batchDetail && (
                <VersionDetailModal
                    detail={batchDetail}
                    onClose={() => { setShowDetailModal(false); setBatchDetail(null); }}
                    onRunNextStep={batchDetail.step_type !== 'reranking' ? (runId, batchId) => {
                        handleRunNextFromDetail(runId, batchId, batchDetail.step_type);
                    } : undefined}
                    nextStepConfig={
                        batchDetail.step_type === 'retrieval' ? { topK, setTopK, model, setModel } :
                            batchDetail.step_type === 'summarization' ? { nSample, setNSample } :
                                undefined
                    }
                    loading={loading}
                />
            )}

            {/* Run Detail Modal — Conversation List */}
            {showRunDetailModal && selectedRun && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => { setShowRunDetailModal(false); setRunConversations([]); }}>
                    <div className="bg-white rounded-xl shadow-2xl w-[92vw] max-w-3xl flex flex-col max-h-[85vh]" onClick={e => e.stopPropagation()}>
                        {/* Header */}
                        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center shrink-0">
                            <div>
                                <h3 className="text-lg font-semibold text-gray-800">
                                    {selectedRun.name ? selectedRun.name : `Run #${selectedRun.id}`}
                                    {selectedRun.name && <span className="ml-2 text-sm font-normal text-gray-400">Run #{selectedRun.id}</span>}
                                </h3>
                                <p className="text-sm text-gray-500 mt-0.5">
                                    {selectedRun.dataset} · {selectedRun.sample_size} conversations
                                </p>
                            </div>
                            <button onClick={() => { setShowRunDetailModal(false); setRunConversations([]); }} className="text-gray-400 hover:text-gray-600 text-2xl leading-none">&times;</button>
                        </div>

                        {/* Body */}
                        <div className="overflow-y-auto flex-1 px-6 py-4">
                            {runDetailLoading ? (
                                <div className="flex items-center justify-center py-16 gap-3">
                                    <svg className="animate-spin h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
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
                                        <div key={conv.id} className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                                            <div className="flex justify-between items-center mb-1">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-xs font-mono text-gray-400">#{idx + 1}</span>
                                                    <span className="text-sm font-medium text-gray-800 truncate max-w-[200px]">{conv.conv_id}</span>
                                                    <span className={`text-xs px-1.5 py-0.5 rounded-full capitalize ${conv.status === 'completed' ? 'bg-green-100 text-green-700' :
                                                        conv.status === 'pending' ? 'bg-yellow-100 text-yellow-700' :
                                                            'bg-gray-100 text-gray-600'
                                                        }`}>{conv.status}</span>
                                                </div>
                                                <span className="text-xs text-gray-500 shrink-0">
                                                    Target: <strong>{Array.isArray(conv.target) ? conv.target.join(', ') : conv.target}</strong>
                                                </span>
                                            </div>
                                            {conv.liked_movies && conv.liked_movies.length > 0 && (
                                                <p className="text-xs text-blue-600 mb-1">❤️ {conv.liked_movies.join(', ')}</p>
                                            )}
                                            {conv.dialog_preview && (
                                                <p className="text-xs text-gray-500 line-clamp-2">{conv.dialog_preview}…</p>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Footer */}
                        <div className="px-6 py-3 border-t border-gray-200 flex justify-end shrink-0">
                            <button
                                onClick={() => { setShowRunDetailModal(false); setRunConversations([]); }}
                                className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
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
                    submitColor="bg-blue-600 hover:bg-blue-700"
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
                                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 border p-2 text-sm"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Dataset</label>
                            <select
                                value={dataset}
                                onChange={(e) => setDataset(e.target.value as "redial" | "inspired")}
                                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 border p-2"
                            >
                                <option value="redial">Redial</option>
                                <option value="inspired">Inspired</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Sample Size</label>
                            <input
                                type="number"
                                value={sampleSize}
                                onChange={(e) => setSampleSize(Number(e.target.value))}
                                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 border p-2"
                            />
                        </div>
                    </div>
                </ModalBackdrop>
            )}

            {/* ========================================= */}
            {/* NEW VERSION MODALS FOR EACH STEP          */}
            {/* ========================================= */}

            {/* New Profiler Agent Version Modal */}
            {showNewVersionModal === 'summ' && (
                <ModalBackdrop
                    title="🤖 New Profiler Agent Version"
                    onClose={() => setShowNewVersionModal(null)}
                    onSubmit={() => handleCreateProfiler()}
                    submitLabel={loading ? 'Running...' : 'Run Profiler Agent'}
                    submitColor="bg-indigo-600 hover:bg-indigo-700"
                    submitDisabled={loading || !selectedRunId}
                >
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Version Name <span className="text-gray-400">(optional)</span></label>
                        <input
                            type="text"
                            value={newVersionName}
                            onChange={(e) => setNewVersionName(e.target.value)}
                            placeholder="e.g. baseline, experiment-1..."
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2 text-sm"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Select Run <span className="text-red-500">*</span></label>
                        <select
                            value={selectedRunId || ''}
                            onChange={(e) => setSelectedRunId(e.target.value ? Number(e.target.value) : undefined)}
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2 text-sm"
                        >
                            <option value="">— Select a run —</option>
                            {allRuns.map(r => (
                                <option key={r.id} value={r.id}>
                                    {r.name ? r.name : `Run #${r.id}`} — {r.dataset} ({r.sample_size} samples) [{r.status}]
                                </option>
                            ))}
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Profiler Agent will extract user preferences from all conversations in this run.</p>
                    </div>
                    <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-3 text-sm text-indigo-800">
                        <p className="font-medium mb-1">ℹ️ No additional parameters needed</p>
                        <p className="text-indigo-600 text-xs">Extracts preferences, genres, hard constraints, and WRRF weights from conversation history.</p>
                    </div>
                </ModalBackdrop>
            )}

            {/* New Retrieval Version Modal */}
            {showNewVersionModal === 'retr' && (
                <ModalBackdrop
                    title="🔍 New Retrieval Version"
                    onClose={() => setShowNewVersionModal(null)}
                    onSubmit={() => {
                        // Infer run_id from selected summarization batch
                        const runId = getRunIdFromSummBatch(selectedSummBatch) || selectedRunId;
                        if (!runId) return;
                        setSelectedRunId(runId);
                        handleCreateRetrieval(runId);
                    }}
                    submitLabel={loading ? 'Running...' : 'Run Retrieval'}
                    submitColor="bg-purple-600 hover:bg-purple-700"
                    submitDisabled={loading || !selectedSummBatch}
                >
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Version Name <span className="text-gray-400">(optional)</span></label>
                        <input
                            type="text"
                            value={newVersionName}
                            onChange={(e) => setNewVersionName(e.target.value)}
                            placeholder="e.g. n100-baseline..."
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2 text-sm"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Input Summarization Version <span className="text-red-500">*</span></label>
                        <select
                            value={selectedSummBatch || ''}
                            onChange={(e) => {
                                const batchId = e.target.value ? Number(e.target.value) : undefined;
                                setSelectedSummBatch(batchId);
                                // Auto-set run_id from the selected batch
                                if (batchId) {
                                    const batch = summBatches.find(b => b.id === batchId);
                                    if (batch) setSelectedRunId(batch.run_id);
                                }
                            }}
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2 text-sm"
                        >
                            <option value="">— Select a summarization version —</option>
                            {summBatches.map(b => (
                                <option key={b.id} value={b.id}>
                                    {b.name ? b.name : `v${b.version}`} (Run #{b.run_id}) — {b.status} — {new Date(b.created_at).toLocaleString()}
                                </option>
                            ))}
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Select which Profiler Agent version to use as input for retrieval.</p>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">N Candidates</label>
                        <input
                            type="number"
                            value={nSample}
                            onChange={(e) => setNSample(Number(e.target.value))}
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2"
                        />
                        <p className="text-xs text-gray-500 mt-1">Number of candidate movies to retrieve (10-600).</p>
                    </div>
                </ModalBackdrop>
            )}

            {/* New Critic + Reranker Version Modal */}
            {showNewVersionModal === 'rerank' && (
                <ModalBackdrop
                    title="🏆 New Critic + Reranker Version"
                    onClose={() => setShowNewVersionModal(null)}
                    onSubmit={() => {
                        const runId = getRunIdFromRetrBatch(selectedRetrBatch) || selectedRunId;
                        if (!runId) return;
                        setSelectedRunId(runId);
                        handleCreateReranking(runId);
                    }}
                    submitLabel={loading ? 'Running...' : 'Run Critic + Reranker'}
                    submitColor="bg-green-600 hover:bg-green-700"
                    submitDisabled={loading || !selectedRetrBatch}
                >
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Version Name <span className="text-gray-400">(optional)</span></label>
                        <input
                            type="text"
                            value={newVersionName}
                            onChange={(e) => setNewVersionName(e.target.value)}
                            placeholder="e.g. cohere-top10..."
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500 border p-2 text-sm"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Input Retrieval Version <span className="text-red-500">*</span></label>
                        <select
                            value={selectedRetrBatch || ''}
                            onChange={(e) => {
                                const batchId = e.target.value ? Number(e.target.value) : undefined;
                                setSelectedRetrBatch(batchId);
                                if (batchId) {
                                    const batch = retrBatches.find(b => b.id === batchId);
                                    if (batch) setSelectedRunId(batch.run_id);
                                }
                            }}
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500 border p-2 text-sm"
                        >
                            <option value="">— Select a retrieval version —</option>
                            {retrBatches.map(b => (
                                <option key={b.id} value={b.id}>
                                    {b.name ? b.name : `v${b.version}`} (Run #{b.run_id}) — N={(b.config as Record<string, unknown>).n_sample as number} — {b.status}
                                </option>
                            ))}
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Select which retrieval version to use as input for reranking.</p>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Top K</label>
                            <input
                                type="number"
                                value={topK}
                                onChange={(e) => setTopK(Number(e.target.value))}
                                className="w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500 border p-2"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
                            <select
                                value={model}
                                onChange={(e) => setModel(e.target.value as "llm" | "cohere")}
                                className="w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500 border p-2"
                            >
                                <option value="cohere">Cohere</option>
                                <option value="llm">LLM</option>
                            </select>
                        </div>
                    </div>
                </ModalBackdrop>
            )}
        </div>
    );
}
