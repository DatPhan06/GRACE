import { useState, useEffect, useCallback } from 'react';
import { Plus } from 'lucide-react';
import Modal from '@/components/ui/Modal';
import RunEvaluationForm from '@/components/evaluation/RunEvaluationForm';
import RunHistoryList from '@/components/evaluation/RunHistoryList';
import RunDetailView from '@/components/evaluation/RunDetailView';
import { getEvaluationRuns, getEvaluationRun, EvaluationRunResponse } from '@/lib/api';

export default function EvaluationPage() {
    const [runs, setRuns] = useState<EvaluationRunResponse[]>([]);
    const [selectedRun, setSelectedRun] = useState<EvaluationRunResponse | null>(null);
    const [isStartModalOpen, setIsStartModalOpen] = useState(false);
    const [loadingRuns, setLoadingRuns] = useState(false);
    const [loadingDetail, setLoadingDetail] = useState(false);

    const STEP_STATUSES = new Set(['initialized', 'summarized', 'retrieved']);

    const fetchRuns = useCallback(async () => {
        setLoadingRuns(true);
        try {
            const data = await getEvaluationRuns();
            setRuns(data.filter(r => !STEP_STATUSES.has(r.status ?? '')));
        } catch (error) {
            console.error('Failed to fetch runs', error);
        } finally {
            setLoadingRuns(false);
        }
    }, []);

    useEffect(() => { fetchRuns(); }, [fetchRuns]);

    useEffect(() => {
        const hasRunning = runs.some(r => r.status === 'running');
        if (!hasRunning) return;
        const interval = setInterval(async () => {
            try {
                const data = await getEvaluationRuns();
                setRuns(data.filter(r => !STEP_STATUSES.has(r.status ?? '')));
            } catch { /* silent */ }
        }, 3000);
        return () => clearInterval(interval);
    }, [runs]);

    const handleSelectRun = async (runId: number) => {
        setLoadingDetail(true);
        try {
            const detail = await getEvaluationRun(runId);
            setSelectedRun(detail);
        } catch (error) {
            console.error('Failed to fetch run details', error);
        } finally {
            setLoadingDetail(false);
        }
    };

    const handleRunComplete = () => {
        setIsStartModalOpen(false);
        fetchRuns();
    };

    const handleDeleteRun = (runId: number) => {
        setRuns(prev => prev.filter(r => r.id !== runId));
        if (selectedRun?.id === runId) setSelectedRun(null);
    };

    return (
        <div className="min-h-full bg-gray-50">
            <div className="max-w-5xl mx-auto px-6 py-8">

                {/* Header */}
                <div className="mb-8 flex items-start justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900 mb-1">System Evaluation</h1>
                        <p className="text-sm text-gray-500">Run end-to-end evaluations and compare model performance.</p>
                    </div>
                    <button
                        onClick={() => setIsStartModalOpen(true)}
                        className="flex items-center gap-2 bg-gray-900 text-white px-4 py-2.5 rounded-xl hover:bg-gray-800 transition-colors text-sm font-medium shadow-sm shrink-0"
                    >
                        <Plus size={16} />
                        New Evaluation
                    </button>
                </div>

                {/* Run history card */}
                <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
                    <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
                        <div>
                            <h2 className="text-sm font-semibold text-gray-800">Run History</h2>
                            <p className="text-xs text-gray-400 mt-0.5">
                                {runs.length} evaluation{runs.length !== 1 ? 's' : ''}
                            </p>
                        </div>
                        <button
                            onClick={fetchRuns}
                            className="text-xs text-gray-400 hover:text-gray-600 transition-colors px-3 py-1.5 rounded-lg hover:bg-gray-50 font-medium"
                        >
                            Refresh
                        </button>
                    </div>

                    {loadingRuns ? (
                        <div className="flex justify-center py-16">
                            <div className="animate-spin h-6 w-6 rounded-full border-2 border-gray-200 border-t-gray-600" />
                        </div>
                    ) : (
                        <RunHistoryList
                            runs={runs}
                            onSelectRun={handleSelectRun}
                            onDeleteRun={handleDeleteRun}
                            activeRunId={selectedRun?.id || null}
                        />
                    )}
                </div>

            </div>

            {/* New Evaluation Modal */}
            <Modal
                isOpen={isStartModalOpen}
                onClose={() => setIsStartModalOpen(false)}
                title="New Evaluation"
                maxWidth="max-w-xl"
            >
                <p className="text-sm text-gray-500 mb-5">
                    Configure the parameters below to launch a new end-to-end evaluation run.
                </p>
                <RunEvaluationForm onRunComplete={handleRunComplete} existingRuns={runs} />
            </Modal>

            {/* Run Detail Modal */}
            <Modal
                isOpen={!!selectedRun}
                onClose={() => setSelectedRun(null)}
                title={selectedRun ? (selectedRun.name || `Run #${selectedRun.id}`) : 'Details'}
                maxWidth="max-w-5xl"
            >
                {selectedRun ? (
                    loadingDetail ? (
                        <div className="flex justify-center py-16">
                            <div className="animate-spin h-6 w-6 rounded-full border-2 border-gray-200 border-t-gray-600" />
                        </div>
                    ) : (
                        <RunDetailView run={selectedRun} />
                    )
                ) : null}
            </Modal>
        </div>
    );
}
