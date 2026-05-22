import { useState, useEffect, useCallback } from 'react';
import { initializeRun, runProfilerStep, runRetrievalStep, runRerankingStep, getStepsByType, getBatchDetail, getEvaluationRuns, getEvaluationInfo, getRunConversations, BatchStepExecutionResponse, BatchDetailResponse, EvaluationRunResponse, ConversationLogItem } from '@/lib/api';
import VersionDetailModal from '@/components/evaluation/VersionDetailModal';

// ─── Shared UI components (defined OUTSIDE the page to keep stable references) ───

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
    <div className="text-center py-12">
        <div className="text-4xl mb-3">{icon}</div>
        <p className="text-gray-500 text-sm">{message}</p>
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

// ─── Pipeline Flow Diagram ─────────────────────────────────────────────────────

function Arrow() {
    return (
        <div className="flex items-center self-start mt-8 shrink-0">
            <div className="w-5 h-0.5 bg-gray-300" />
            <div className="w-0 h-0" style={{ borderTop: '5px solid transparent', borderBottom: '5px solid transparent', borderLeft: '7px solid #d1d5db' }} />
        </div>
    );
}

function PipelineFlowDiagram() {
    return (
        <div className="bg-white border border-gray-200 rounded-xl px-6 py-5 mb-8 shadow-sm">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-5">ARGOS Evaluation Pipeline</p>
            <div className="flex items-start gap-0 overflow-x-auto pb-2">

                {/* Step 1: Initialize */}
                <div className="flex flex-col items-center shrink-0">
                    <div className="w-28 p-3 bg-blue-50 border-2 border-blue-200 rounded-xl text-center">
                        <div className="text-xl mb-1">🚀</div>
                        <div className="text-xs font-bold text-blue-800">Initialize</div>
                        <div className="text-[10px] text-blue-500 mt-0.5">Dataset sample</div>
                    </div>
                </div>
                <Arrow />

                {/* Step 2: Profiler Agent */}
                <div className="flex flex-col items-center shrink-0">
                    <div className="w-28 p-3 bg-indigo-50 border-2 border-indigo-200 rounded-xl text-center">
                        <div className="text-xl mb-1">🤖</div>
                        <div className="text-xs font-bold text-indigo-800">Profiler Agent</div>
                        <div className="text-[10px] text-indigo-500 mt-0.5">Extract prefs</div>
                    </div>
                </div>
                <Arrow />

                {/* Step 3: Retrieval + Critic + Reflexion */}
                <div className="flex flex-col items-center shrink-0">
                    <div className="border-2 border-purple-200 bg-purple-50 rounded-xl p-3 w-56">
                        <div className="text-[10px] font-bold text-purple-600 uppercase tracking-wider mb-2 text-center">
                            Retrieval + Critic + Reflexion
                        </div>
                        <div className="flex flex-col items-center gap-1">
                            <div className="bg-white border border-purple-200 rounded-lg px-3 py-1.5 text-[11px] font-semibold text-purple-700 w-full text-center">
                                Retrieval (WRRF fusion)
                            </div>
                            <div className="text-[10px] text-gray-400">↓</div>
                            <div className="bg-white border border-orange-200 rounded-lg px-3 py-1.5 text-[11px] font-semibold text-orange-700 w-full text-center">
                                Critic Agent
                            </div>
                            <div className="text-[10px] text-gray-400">↓</div>
                            {/* Reflexion loop — dashed signals conditional re-retrieval */}
                            <div className="w-full border border-dashed border-amber-300 bg-amber-50/60 rounded-lg px-2.5 py-1.5 text-center">
                                <div className="text-[9px] font-semibold text-amber-600 mb-0.5">if |M| &lt; τ · max 1×</div>
                                <div className="text-[10px] text-amber-700">Reflexion → re-retrieve → Critic</div>
                            </div>
                            <div className="text-[10px] text-gray-400 italic">↓ always</div>
                            <div className="bg-white border border-purple-100 rounded-lg px-3 py-1.5 text-[11px] text-purple-600 w-full text-center">
                                filtered candidates
                            </div>
                        </div>
                    </div>
                </div>
                <Arrow />

                {/* Step 4: Reranker */}
                <div className="flex flex-col items-center shrink-0">
                    <div className="w-32 p-3 bg-green-50 border-2 border-green-200 rounded-xl text-center">
                        <div className="text-xl mb-1">🎯</div>
                        <div className="text-xs font-bold text-green-800">Reranker</div>
                        <div className="text-[10px] text-green-500 mt-0.5">Recall@K</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// ─── Section wrapper ───────────────────────────────────────────────────────────

function PipelineSection({ id, stepNumber, title, subtitle, accentColor, headerRight, children }: {
    id: string;
    stepNumber: string;
    title: string;
    subtitle: string;
    accentColor: string;
    headerRight?: React.ReactNode;
    children: React.ReactNode;
}) {
    return (
        <section id={id} className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden mb-6">
            <div className={`px-6 py-4 border-b border-gray-100 flex justify-between items-start ${accentColor}`}>
                <div>
                    <div className="flex items-center gap-2 mb-0.5">
                        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{stepNumber}</span>
                        <h2 className="text-lg font-bold text-gray-800">{title}</h2>
                    </div>
                    <p className="text-sm text-gray-500">{subtitle}</p>
                </div>
                {headerRight}
            </div>
            <div className="px-6 py-5">
                {children}
            </div>
        </section>
    );
}

// ─── Main Page ─────────────────────────────────────────────────────────────────

export default function StepEvaluationPage() {
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

    // New Version Modal — 'summ' | 'retrieval' | 'rerank' | null
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

    // Load all on mount
    useEffect(() => {
        loadRuns();
        loadSummBatches();
        loadRetrievalBatches();
        loadRerankBatches();
        getEvaluationInfo().then(info => setDatasetSizes(info.dataset_sizes)).catch(() => {});
    }, [loadRuns, loadSummBatches, loadRetrievalBatches, loadRerankBatches]);

    // Step-based runs only: 'initialized' (created via initializeRun, no steps yet)
    // or referenced in any batch step.
    const stepRunIds = new Set([
        ...summBatches.map(b => b.run_id),
        ...retrievalBatches.map(b => b.run_id),
        ...rerankBatches.map(b => b.run_id),
    ]);
    const stepRuns = allRuns.filter(r => r.status === 'initialized' || stepRunIds.has(r.id));

    // Polling: watch any running batch
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

    // --- Open New Version Modal ---
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

    // --- Handlers ---
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

    // Step 3: Run Retrieval + Critic + Reflexion
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

    // Step 4: Run Reranker (pure reranking, using filtered_candidates from Step 3)
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

    // "Run Next Step" from detail modal: summarization → run retrieval; retrieval → handled by modal directly
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

    return (
        <div className="container mx-auto px-4 py-8 max-w-4xl">
            <h1 className="text-3xl font-bold mb-2">Step-by-Step Optimization</h1>
            <p className="text-gray-500 text-sm mb-8">
                Run each pipeline stage independently: Profiler Agent extracts preferences, then
                Retrieval+Critic+Reflexion retrieves and filters candidates, then Reranker selects top-K.
            </p>

            {/* Pipeline Diagram */}
            <PipelineFlowDiagram />

            {/* ─── Section 1: Evaluation Runs ─── */}
            <PipelineSection
                id="section-init"
                stepNumber="Step 1"
                title="Evaluation Runs"
                subtitle="Create runs with dataset samples to use across all pipeline steps."
                accentColor="bg-blue-50/50"
                headerRight={
                    <button
                        onClick={() => { setNewVersionName(''); setShowRunModal(true); }}
                        className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium shadow-sm"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
                        New Run
                    </button>
                }
            >
                <div className="space-y-3">
                    {stepRuns.length === 0 && <EmptyState icon="🚀" message="No runs yet. Click '+ New Run' to create one." />}
                    {stepRuns.map(run => (
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
                            <div className="flex items-center gap-2">
                                <span className="bg-white px-2 py-0.5 rounded border border-gray-200 text-xs">Dataset: <strong className="capitalize">{run.dataset}</strong></span>
                                <span className="bg-white px-2 py-0.5 rounded border border-gray-200 text-xs">Samples: <strong>{run.sample_size}</strong></span>
                            </div>
                        </div>
                    ))}
                </div>
            </PipelineSection>

            {/* ─── Section 2: Profiler Agent ─── */}
            <PipelineSection
                id="section-profiler"
                stepNumber="Step 2"
                title="Profiler Agent"
                subtitle="Extracts user preferences, genres, hard constraints, and WRRF weights from conversation history."
                accentColor="bg-indigo-50/50"
                headerRight={<NewVersionButton onClick={() => openNewVersionModal('summ')} color="bg-indigo-600 hover:bg-indigo-700" />}
            >
                <div className="space-y-3">
                    {summBatches.length === 0 && <EmptyState icon="🤖" message="No Profiler Agent versions yet. Click '+ New Version' to create one." />}
                    {summBatches.map(b => (
                        <VersionCard key={b.id} batch={b} color={{ bg: 'bg-indigo-50', border: 'border-indigo-100', text: 'text-indigo-800' }} onClick={() => handleViewDetail(b)} />
                    ))}
                </div>
            </PipelineSection>

            {/* ─── Section 3: Retrieval + Critic + Reflexion ─── */}
            <PipelineSection
                id="section-retrieval"
                stepNumber="Step 3"
                title="Retrieval + Critic + Reflexion"
                subtitle="WRRF retrieval, then Critic filters candidates. If too few pass constraints, Reflexion widens them and re-retrieves (max 1x)."
                accentColor="bg-purple-50/50"
                headerRight={<NewVersionButton onClick={() => openNewVersionModal('retrieval')} color="bg-purple-600 hover:bg-purple-700" />}
            >
                <div className="space-y-3">
                    {retrievalBatches.length === 0 && <EmptyState icon="🔍" message="No Retrieval+Critic versions yet. Click '+ New Version' to create one." />}
                    {retrievalBatches.map(b => (
                        <VersionCard key={b.id} batch={b} color={{ bg: 'bg-purple-50', border: 'border-purple-100', text: 'text-purple-800' }} onClick={() => handleViewDetail(b)}>
                            <div className="flex gap-3 text-xs text-gray-600">
                                <span>N={(b.config as Record<string, unknown>).n_sample as number}</span>
                            </div>
                        </VersionCard>
                    ))}
                </div>
            </PipelineSection>

            {/* ─── Section 4: Reranker → Recall@K ─── */}
            <PipelineSection
                id="section-reranker"
                stepNumber="Step 4"
                title="Reranker → Recall@K"
                subtitle="Pure reranking on post-Critic filtered candidates from Step 3. Computes Recall@K."
                accentColor="bg-green-50/50"
                headerRight={<NewVersionButton onClick={() => openNewVersionModal('rerank')} color="bg-green-600 hover:bg-green-700" />}
            >
                <div className="space-y-3">
                    {rerankBatches.length === 0 && <EmptyState icon="🎯" message="No Reranker versions yet. Click '+ New Version' to create one." />}
                    {rerankBatches.map(b => (
                        <VersionCard key={b.id} batch={b} color={{ bg: 'bg-green-50', border: 'border-green-100', text: 'text-green-800' }} onClick={() => handleViewDetail(b)}>
                            <div className="flex gap-3 text-xs text-gray-600">
                                <span>TopK={(b.config as Record<string, unknown>).top_k as number}</span>
                                <span>{String((b.config as Record<string, unknown>).model)}</span>
                            </div>
                        </VersionCard>
                    ))}
                </div>
            </PipelineSection>

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
                    <div className="bg-white rounded-xl shadow-2xl w-[92vw] max-w-3xl flex flex-col max-h-[85vh]" onClick={e => e.stopPropagation()}>
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
                                                    <span className={`text-xs px-1.5 py-0.5 rounded-full capitalize ${conv.status === 'completed' ? 'bg-green-100 text-green-700' : conv.status === 'pending' ? 'bg-yellow-100 text-yellow-700' : 'bg-gray-100 text-gray-600'}`}>
                                                        {conv.status}
                                                    </span>
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
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Sample Size
                                <span className="ml-2 font-semibold text-blue-600">{samplePercent}%</span>
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
                                className="w-full accent-blue-600"
                            />
                            <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                                <span>1%</span>
                                <span>50%</span>
                                <span>100%</span>
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
                            {stepRuns.map(r => (
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

            {/* New Retrieval+Critic+Reflexion Version Modal */}
            {showNewVersionModal === 'retrieval' && (
                <ModalBackdrop
                    title="New Retrieval + Critic + Reflexion Version"
                    onClose={() => setShowNewVersionModal(null)}
                    onSubmit={() => {
                        const runId = getRunIdFromSummBatch(selectedSummBatch) || selectedRunId;
                        if (!runId) return;
                        setSelectedRunId(runId);
                        handleRunRetrieval(selectedSummBatch);
                    }}
                    submitLabel={loading ? 'Running...' : 'Run Retrieval + Critic + Reflexion'}
                    submitColor="bg-purple-600 hover:bg-purple-700"
                    submitDisabled={loading || !selectedSummBatch}
                >
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Version Name <span className="text-gray-400">(optional)</span></label>
                        <input
                            type="text"
                            value={newVersionName}
                            onChange={(e) => setNewVersionName(e.target.value)}
                            placeholder="e.g. n100-critic-v1..."
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2 text-sm"
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
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2 text-sm"
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
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2"
                        />
                        <p className="text-xs text-gray-500 mt-1">Number of candidates to retrieve (10–600).</p>
                    </div>
                    <div className="bg-purple-50 border border-purple-100 rounded-lg p-3 text-xs text-purple-800">
                        Retrieval runs first (WRRF fusion), then Critic checks constraints. If relaxation is needed, it re-retrieves once before producing filtered candidates.
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
                    submitColor="bg-green-600 hover:bg-green-700"
                    submitDisabled={loading || !selectedRetrievalBatch}
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
                            value={selectedRetrievalBatch || ''}
                            onChange={(e) => {
                                const batchId = e.target.value ? Number(e.target.value) : undefined;
                                setSelectedRetrievalBatch(batchId);
                                if (batchId) {
                                    const batch = retrievalBatches.find(b => b.id === batchId);
                                    if (batch) setSelectedRunId(batch.run_id);
                                }
                            }}
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500 border p-2 text-sm"
                        >
                            <option value="">— Select a retrieval version —</option>
                            {retrievalBatches.filter(b => b.status === 'completed').map(b => (
                                <option key={b.id} value={b.id}>
                                    {b.name ? b.name : `v${b.version}`} (Run #{b.run_id}) — {new Date(b.created_at).toLocaleString()}
                                </option>
                            ))}
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Only completed retrieval versions are shown.</p>
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
                    <div className="bg-green-50 border border-green-100 rounded-lg p-3 text-xs text-green-800">
                        Uses filtered candidates from the selected Retrieval+Critic+Reflexion version (Step 3).
                    </div>
                </ModalBackdrop>
            )}
        </div>
    );
}
