import { EvaluationRunResponse } from '@/lib/api';

interface RunHistoryListProps {
    runs: EvaluationRunResponse[];
    onSelectRun: (runId: number) => void;
    activeRunId: number | null;
}

export default function RunHistoryList({ runs, onSelectRun, activeRunId }: RunHistoryListProps) {
    if (runs.length === 0) {
        return <div className="text-gray-500 text-center py-4">No history available</div>;
    }

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-sm text-left text-gray-600">
                <thead className="bg-gray-100 text-gray-700 uppercase text-xs">
                    <tr>
                        <th className="px-4 py-3">ID</th>
                        <th className="px-4 py-3">Dataset</th>
                        <th className="px-4 py-3">Model</th>
                        <th className="px-4 py-3">Size</th>
                        <th className="px-4 py-3">Recall@K</th>
                        <th className="px-4 py-3">Retrieval@N</th>
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
                            <td className="px-4 py-3 font-medium text-gray-900">#{run.id}</td>
                            <td className="px-4 py-3">{run.dataset}</td>
                            <td className="px-4 py-3 uppercase text-xs font-bold text-gray-500">{run.model}</td>
                            <td className="px-4 py-3">{run.sample_size}</td>
                            <td className="px-4 py-3 font-semibold text-blue-600">
                                {run.avg_recall?.toFixed(4)}
                            </td>
                            <td className="px-4 py-3 text-gray-500">
                                {run.avg_recall_retrieval?.toFixed(4) || "N/A"}
                            </td>
                            <td className="px-4 py-3 text-xs text-gray-500">
                                {run.timestamp ? new Date(run.timestamp).toLocaleString() : 'N/A'}
                            </td>
                            <td className="px-4 py-3">
                                <button
                                    onClick={() => onSelectRun(run.id)}
                                    className="text-blue-600 hover:text-blue-800 font-medium hover:underline"
                                >
                                    View
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
