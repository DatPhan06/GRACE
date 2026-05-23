import { useState } from 'react';
import { Trash2, BarChart2 } from 'lucide-react';
import { EvaluationRunResponse, deleteEvaluationRun } from '@/lib/api';

interface RunHistoryListProps {
    runs: EvaluationRunResponse[];
    onSelectRun: (runId: number) => void;
    onDeleteRun: (runId: number) => void;
    activeRunId: number | null;
}

function StatusBadge({ status }: { status: string }) {
    if (status === 'running') return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-100 text-amber-700">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
            Running
        </span>
    );
    if (status === 'failed') return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-red-100 text-red-700">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
            Failed
        </span>
    );
    return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-green-100 text-green-700">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
            Done
        </span>
    );
}

export default function RunHistoryList({ runs, onSelectRun, onDeleteRun, activeRunId }: RunHistoryListProps) {
    const [deletingId, setDeletingId] = useState<number | null>(null);

    const handleDelete = async (e: React.MouseEvent, run: EvaluationRunResponse) => {
        e.stopPropagation();
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
        return (
            <div className="flex flex-col items-center justify-center py-16 text-center">
                <BarChart2 size={32} className="text-gray-200 mb-3" />
                <p className="text-gray-400 text-sm">No evaluation runs yet.</p>
                <p className="text-gray-300 text-xs mt-1">Click "New Evaluation" to get started.</p>
            </div>
        );
    }

    return (
        <div className="px-4 py-3 space-y-2.5">
            {runs.map(run => {
                const isActive = activeRunId === run.id;
                const isRunning = run.status === 'running';
                return (
                    <div
                        key={run.id}
                        onClick={() => !isRunning && onSelectRun(run.id)}
                        className={`p-4 rounded-2xl border transition-all group ${
                            isActive
                                ? 'border-gray-300 bg-gray-50 shadow-sm'
                                : isRunning
                                    ? 'border-gray-200 bg-white cursor-default'
                                    : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm cursor-pointer'
                        }`}
                    >
                        {/* Row 1: name + status + actions */}
                        <div className="flex items-center justify-between mb-2.5">
                            <div className="flex items-center gap-2 min-w-0">
                                <span className="font-semibold text-gray-900 text-sm truncate">
                                    {run.name || `Run #${run.id}`}
                                </span>
                                {run.name && <span className="text-xs text-gray-400 shrink-0">#{run.id}</span>}
                                <StatusBadge status={run.status ?? 'done'} />
                            </div>
                            <div className="flex items-center gap-2 shrink-0 ml-2">
                                <span className="text-xs text-gray-400 hidden sm:block">
                                    {run.timestamp ? new Date(run.timestamp).toLocaleString() : ''}
                                </span>
                                <button
                                    onClick={e => handleDelete(e, run)}
                                    disabled={deletingId === run.id}
                                    className="p-1 rounded-lg opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 hover:bg-red-50 transition-all disabled:opacity-30"
                                    title="Delete run"
                                >
                                    <Trash2 size={13} />
                                </button>
                            </div>
                        </div>

                        {/* Row 2: tags left, recall right */}
                        <div className="flex items-center justify-between gap-4">
                            <div className="flex flex-wrap gap-1.5 items-center min-w-0">
                                <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-lg capitalize">{run.dataset}</span>
                                <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-lg uppercase font-mono">{run.model}</span>
                                <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-lg">{run.sample_size} samples</span>
                                {run.top_k != null && <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-lg">Top-{run.top_k}</span>}
                            </div>
                            {run.avg_recall != null && !isRunning ? (
                                <div className="shrink-0 text-right">
                                    <div className="text-[10px] text-gray-400 font-medium uppercase tracking-wider leading-none mb-0.5">Recall@{run.top_k}</div>
                                    <div className="text-lg font-bold text-blue-600 leading-none">{run.avg_recall.toFixed(4)}</div>
                                </div>
                            ) : isRunning ? (
                                <div className="shrink-0 text-[11px] text-gray-400 italic">computing…</div>
                            ) : null}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
