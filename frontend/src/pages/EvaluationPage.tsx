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

    // Loading states
    const [loadingRuns, setLoadingRuns] = useState(false);
    const [loadingDetail, setLoadingDetail] = useState(false);

    const STEP_STATUSES = new Set(['initialized', 'summarized', 'retrieved']);

    const fetchRuns = useCallback(async () => {
        setLoadingRuns(true);
        try {
            const data = await getEvaluationRuns();
            setRuns(data.filter(r => !STEP_STATUSES.has(r.status ?? '')));
        } catch (error) {
            console.error("Failed to fetch runs", error);
        } finally {
            setLoadingRuns(false);
        }
    }, []);

    useEffect(() => {
        fetchRuns();
    }, [fetchRuns]);

    // Poll while any run is still processing
    useEffect(() => {
        const hasRunning = runs.some(r => r.status === 'running');
        if (!hasRunning) return;
        const interval = setInterval(async () => {
            try {
                const data = await getEvaluationRuns();
                setRuns(data);
            } catch {
                // silent — don't disrupt UX on polling failure
            }
        }, 3000);
        return () => clearInterval(interval);
    }, [runs]);

    const handleSelectRun = async (runId: number) => {
        setLoadingDetail(true);
        // Show modal immediately, loading state inside if needed or manage here
        try {
            const detail = await getEvaluationRun(runId);
            setSelectedRun(detail);
        } catch (error) {
            console.error("Failed to fetch run details", error);
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
        <div className="container mx-auto px-4 py-8 max-w-7xl">
            <header className="mb-8 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">System Evaluation</h1>
                    <p className="text-gray-500 mt-2">Manage and analyze model performance experiments.</p>
                </div>
                <button
                    onClick={() => setIsStartModalOpen(true)}
                    className="flex items-center space-x-2 bg-blue-600 text-white px-6 py-2.5 rounded-lg hover:bg-blue-700 transition-colors shadow-sm font-medium"
                >
                    <Plus size={20} />
                    <span>New Evaluation</span>
                </button>
            </header>

            {/* Main Content: Full-width History */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="p-4 border-b border-gray-100 bg-gray-50 flex justify-between items-center">
                    <h3 className="font-semibold text-gray-800">Run History</h3>
                    <button
                        onClick={fetchRuns}
                        className="text-sm text-blue-600 hover:text-blue-800 font-medium"
                    >
                        Refresh List
                    </button>
                </div>

                {loadingRuns ? (
                    <div className="p-8 text-center text-gray-500">Loading history...</div>
                ) : (
                    <RunHistoryList
                        runs={runs}
                        onSelectRun={handleSelectRun}
                        onDeleteRun={handleDeleteRun}
                        activeRunId={selectedRun?.id || null}
                    />
                )}
            </div>

            {/* Modals */}

            {/* 1. Start Evaluation Modal */}
            <Modal
                isOpen={isStartModalOpen}
                onClose={() => setIsStartModalOpen(false)}
                title="Start New Evaluation"
                maxWidth="max-w-2xl"
            >
                <div className="pb-2">
                    <p className="text-gray-500 mb-6 font-light">
                        Configure the parameters below to launch a new evaluation pipeline run.
                        Results will be saved automatically.
                    </p>
                    <RunEvaluationForm onRunComplete={handleRunComplete} existingRuns={runs} />
                </div>
            </Modal>

            {/* 2. Run Details Modal */}
            <Modal
                isOpen={!!selectedRun}
                onClose={() => setSelectedRun(null)}
                title={selectedRun ? `Evaluation Details #${selectedRun.id}` : 'Details'}
                maxWidth="max-w-6xl"
            >
                {/* 
                   We check !!selectedRun for the modal open state, 
                   so selectedRun is guaranteed to be present here render-wise due to React batching,
                   but for strict TS, conditional rendering inside content
                */}
                {selectedRun ? (
                    loadingDetail ? (
                        <div className="flex justify-center p-12">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                        </div>
                    ) : (
                        <div className="-mt-8"> {/* Negative margin to offset inner component spacing if needed */}
                            <RunDetailView run={selectedRun} />
                        </div>
                    )
                ) : null}
            </Modal>
        </div>
    );
}
