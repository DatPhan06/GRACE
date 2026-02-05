import { useState, useEffect, useCallback } from 'react';
import RunEvaluationForm from '@/components/evaluation/RunEvaluationForm';
import RunHistoryList from '@/components/evaluation/RunHistoryList';
import RunDetailView from '@/components/evaluation/RunDetailView';
import { getEvaluationRuns, getEvaluationRun, EvaluationRunResponse } from '@/lib/api';

export default function EvaluationPage() {
    const [runs, setRuns] = useState<EvaluationRunResponse[]>([]);
    const [selectedRun, setSelectedRun] = useState<EvaluationRunResponse | null>(null);
    const [loadingRuns, setLoadingRuns] = useState(false);
    const [loadingDetail, setLoadingDetail] = useState(false);

    const fetchRuns = useCallback(async () => {
        setLoadingRuns(true);
        try {
            const data = await getEvaluationRuns();
            setRuns(data);
            // Select first run by default if none selected
            if (data.length > 0 && !selectedRun) {
                // Optional: Auto-select latest
                // handleSelectRun(data[0].id); 
            }
        } catch (error) {
            console.error("Failed to fetch runs", error);
        } finally {
            setLoadingRuns(false);
        }
    }, [selectedRun]);

    useEffect(() => {
        fetchRuns();
    }, [fetchRuns]);

    const handleSelectRun = async (runId: number) => {
        setLoadingDetail(true);
        try {
            const detail = await getEvaluationRun(runId);
            setSelectedRun(detail);
        } catch (error) {
            console.error("Failed to fetch run details", error);
        } finally {
            setLoadingDetail(false);
        }
    };

    return (
        <div className="container mx-auto px-4 py-8 max-w-6xl">
            <header className="mb-8 border-b pb-4">
                <h1 className="text-3xl font-bold text-gray-900">System Evaluation</h1>
                <p className="text-gray-500 mt-2">Run experiments and analyze model performance.</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Column: Form & History List */}
                <div className="lg:col-span-1 space-y-6">
                    <RunEvaluationForm onRunComplete={fetchRuns} />

                    <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-4">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-lg font-semibold text-gray-800">History</h3>
                            <button
                                onClick={fetchRuns}
                                className="text-sm text-blue-600 hover:underline"
                            >
                                Refresh
                            </button>
                        </div>
                        {loadingRuns ? (
                            <div className="text-center py-4">Loading history...</div>
                        ) : (
                            <RunHistoryList
                                runs={runs}
                                onSelectRun={handleSelectRun}
                                activeRunId={selectedRun?.id || null}
                            />
                        )}
                    </div>
                </div>

                {/* Right Column: Detail View */}
                <div className="lg:col-span-2">
                    {loadingDetail ? (
                        <div className="bg-white p-12 rounded-lg shadow-sm border border-gray-100 flex justify-center">
                            <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                        </div>
                    ) : selectedRun ? (
                        <RunDetailView run={selectedRun} />
                    ) : (
                        <div className="bg-gray-50 rounded-lg border-2 border-dashed border-gray-200 p-12 text-center text-gray-400">
                            Select a run from history to view details
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
