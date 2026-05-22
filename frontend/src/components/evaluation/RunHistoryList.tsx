import { useState } from 'react';
import { Trash2 } from 'lucide-react';
import { EvaluationRunResponse, deleteEvaluationRun } from '@/lib/api';

interface RunHistoryListProps {
    runs: EvaluationRunResponse[];
    onSelectRun: (runId: number) => void;
    onDeleteRun: (runId: number) => void;
    activeRunId: number | null;
}

export default function RunHistoryList({ runs, onSelectRun, onDeleteRun, activeRunId }: RunHistoryListProps) {
    const [deletingId, setDeletingId] = useState<number | null>(null);

    const handleDelete = async (run: EvaluationRunResponse) => {
        const label = run.name ?? `#${run.id}`;
        if (!window.confirm(`Delete "${label}"?\nThis will remove all results permanently.`)) return;
        setDeletingId(run.id);
        try {
            await deleteEvaluationRun(run.id);
            onDeleteRun(run.id);
        } catch {
            alert('Failed to delete run.');
        } finally {
            setDeletingId(null);
        }
    };

    if (runs.length === 0) {
        return <div className="text-gray-500 text-center py-4">No history available</div>;
    }

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-sm text-left text-gray-600">
                <thead className="bg-gray-100 text-gray-700 uppercase text-xs">
                    <tr>
                        <th className="px-4 py-3">ID</th>
                        <th className="px-4 py-3">Status</th>
                        <th className="px-4 py-3">Dataset</th>
                        <th className="px-4 py-3">Model</th>
                        <th className="px-4 py-3">Size</th>
                        <th className="px-4 py-3">Recall@K</th>
                        <th className="px-4 py-3">Date</th>
                        <th className="px-4 py-3">Action</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 bg-white">
                    {runs.map((run) => (
                        <tr
                            key={run.id}
                            className={`hover:bg-gray-50 transition-colors ${activeRunId === run.id ? 'bg-blue-50' : ''}`}
                        >
                            <td className="px-4 py-3 font-medium text-gray-900">
                                {run.name ? (
                                    <span title={`#${run.id}`}>{run.name}</span>
                                ) : (
                                    `#${run.id}`
                                )}
                            </td>
                            <td className="px-4 py-3">
                                {run.status === 'running' ? (
                                    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                                        <span className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse" />
                                        Running
                                    </span>
                                ) : run.status === 'failed' ? (
                                    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">
                                        <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                                        Failed
                                    </span>
                                ) : (
                                    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700">
                                        <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                                        Done
                                    </span>
                                )}
                            </td>
                            <td className="px-4 py-3">{run.dataset}</td>
                            <td className="px-4 py-3 uppercase text-xs font-bold text-gray-500">{run.model}</td>
                            <td className="px-4 py-3">{run.sample_size}</td>
                            <td className="px-4 py-3 font-semibold text-blue-600">
                                {run.status === 'running' ? '—' : run.avg_recall?.toFixed(4)}
                            </td>
                            <td className="px-4 py-3 text-xs text-gray-500">
                                {run.timestamp ? new Date(run.timestamp).toLocaleString() : 'N/A'}
                            </td>
                            <td className="px-4 py-3">
                                <div className="flex items-center gap-3">
                                    <button
                                        onClick={() => onSelectRun(run.id)}
                                        disabled={run.status === 'running'}
                                        className={`font-medium ${run.status === 'running' ? 'text-gray-300 cursor-not-allowed' : 'text-blue-600 hover:text-blue-800 hover:underline'}`}
                                    >
                                        View
                                    </button>
                                    <button
                                        onClick={() => handleDelete(run)}
                                        disabled={deletingId === run.id}
                                        className="text-gray-400 hover:text-red-500 transition-colors disabled:opacity-40"
                                        title="Delete run"
                                    >
                                        <Trash2 size={14} />
                                    </button>
                                </div>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
