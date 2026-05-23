import { useState } from 'react';
import { Trash2, BarChart2, ChevronLeft, ChevronRight } from 'lucide-react';

const PAGE_SIZE = 4;
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
    const [page, setPage] = useState(1);

    const totalPages = Math.max(1, Math.ceil(runs.length / PAGE_SIZE));
    const safePage = Math.min(page, totalPages);
    const paged = runs.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

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
        <div>
        <div className="px-4 py-3 space-y-2.5">
            {paged.map(run => {
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

        {/* Pagination */}
        {totalPages > 1 && (
            <div className="flex items-center justify-between px-5 py-3 border-t border-gray-100">
                <span className="text-xs text-gray-400">
                    {(safePage - 1) * PAGE_SIZE + 1}–{Math.min(safePage * PAGE_SIZE, runs.length)} of {runs.length}
                </span>
                <div className="flex items-center gap-1">
                    <button
                        onClick={() => setPage(p => Math.max(1, p - 1))}
                        disabled={safePage === 1}
                        className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                        <ChevronLeft size={15} />
                    </button>
                    {Array.from({ length: totalPages }, (_, i) => i + 1).map(p => (
                        <button
                            key={p}
                            onClick={() => setPage(p)}
                            className={`w-7 h-7 rounded-lg text-xs font-medium transition-colors ${
                                p === safePage
                                    ? 'bg-gray-900 text-white'
                                    : 'text-gray-500 hover:bg-gray-100'
                            }`}
                        >
                            {p}
                        </button>
                    ))}
                    <button
                        onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                        disabled={safePage === totalPages}
                        className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                        <ChevronRight size={15} />
                    </button>
                </div>
            </div>
        )}
        </div>
    );
}
