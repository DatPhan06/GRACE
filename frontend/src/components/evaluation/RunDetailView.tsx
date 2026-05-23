import { EvaluationRunResponse, EvaluationResultResponse, exportEvaluationRun } from '@/lib/api';
import { useState } from 'react';
import { Download } from 'lucide-react';

const PREVIEW_COUNT = 10;

function RecommendationsList({ recommendations }: { recommendations: EvaluationResultResponse['recommendations'] }) {
    const [expanded, setExpanded] = useState(false);
    const items: string[] = Array.isArray(recommendations)
        ? recommendations.map(String)
        : recommendations ? [String(recommendations)] : [];

    if (items.length === 0) return <p className="font-mono text-xs text-gray-400 mt-1">—</p>;

    const shown = expanded ? items : items.slice(0, PREVIEW_COUNT);
    const hidden = items.length - PREVIEW_COUNT;

    return (
        <div className="mt-1.5">
            <ol className="font-mono text-xs bg-gray-50 border border-gray-200 rounded-xl p-2.5 space-y-0.5">
                {shown.map((title, i) => (
                    <li key={i} className="flex gap-2">
                        <span className="text-gray-300 w-5 shrink-0 text-right">{i + 1}.</span>
                        <span className="break-words text-gray-700">{title}</span>
                    </li>
                ))}
            </ol>
            {hidden > 0 && (
                <button
                    type="button"
                    onClick={() => setExpanded(v => !v)}
                    className="text-xs text-gray-500 hover:text-gray-700 mt-1.5 font-medium"
                >
                    {expanded ? 'Show less' : `+ ${hidden} more`}
                </button>
            )}
        </div>
    );
}

interface RunDetailViewProps {
    run: EvaluationRunResponse;
}

export default function RunDetailView({ run }: RunDetailViewProps) {
    const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
    const [exporting, setExporting] = useState(false);

    const handleExport = async () => {
        setExporting(true);
        try {
            await exportEvaluationRun(run);
        } catch (err) {
            console.error('Export failed', err);
        } finally {
            setExporting(false);
        }
    };

    const toggleRow = (convId: string) => {
        const next = new Set(expandedRows);
        if (next.has(convId)) next.delete(convId);
        else next.add(convId);
        setExpandedRows(next);
    };

    const results = run.results || [];

    return (
        <div>
            {/* Metrics strip */}
            <div className="grid grid-cols-3 gap-3 mb-5">
                <div className="bg-white rounded-2xl border border-gray-200 p-4 text-center">
                    <div className="text-[11px] text-gray-400 font-medium uppercase tracking-wider mb-1">Recall@{run.top_k}</div>
                    <div className="text-2xl font-bold text-blue-600">{run.avg_recall?.toFixed(4) ?? '—'}</div>
                </div>
                <div className="bg-white rounded-2xl border border-gray-200 p-4 text-center">
                    <div className="text-[11px] text-gray-400 font-medium uppercase tracking-wider mb-1">Model</div>
                    <div className="text-lg font-bold text-gray-800 uppercase">{run.model ?? '—'}</div>
                    {run.llm_model && <div className="text-[10px] text-gray-400 mt-0.5">{run.llm_model}</div>}
                </div>
                <div className="bg-white rounded-2xl border border-gray-200 p-4 text-center">
                    <div className="text-[11px] text-gray-400 font-medium uppercase tracking-wider mb-1">Config</div>
                    <div className="text-base font-bold text-gray-800">Top-{run.top_k} / N={run.n_sample}</div>
                    <div className="text-[10px] text-gray-400 mt-0.5">{run.sample_size} conversations</div>
                </div>
            </div>

            {/* Export + table header */}
            <div className="flex items-center justify-between mb-3">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    Per-conversation results ({results.length})
                </p>
                <button
                    onClick={handleExport}
                    disabled={exporting || results.length === 0}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium border border-gray-200 bg-white text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                    <Download size={13} />
                    {exporting ? 'Exporting…' : 'Export TSV'}
                </button>
            </div>

            {/* Per-conversation table */}
            <div className="rounded-2xl border border-gray-200 overflow-hidden">
                <table className="w-full text-sm text-left">
                    <thead className="bg-gray-50 border-b border-gray-200">
                        <tr>
                            <th className="px-4 py-3 w-4"></th>
                            <th className="px-4 py-3 text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Conv ID</th>
                            <th className="px-4 py-3 text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Recall@K</th>
                            <th className="px-4 py-3 text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Candidates</th>
                            <th className="px-4 py-3 text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                        {results.map((res: EvaluationResultResponse) => (
                            <>
                                <tr
                                    key={res.conv_id}
                                    onClick={() => toggleRow(res.conv_id)}
                                    className="hover:bg-gray-50/80 cursor-pointer transition-colors"
                                >
                                    <td className="px-4 py-3 text-gray-300 text-xs">
                                        {expandedRows.has(res.conv_id) ? '▼' : '▶'}
                                    </td>
                                    <td className="px-4 py-3 font-mono text-xs text-gray-700 max-w-[180px] truncate">
                                        {res.conv_id}
                                    </td>
                                    <td className="px-4 py-3 font-semibold text-blue-600 text-sm">
                                        {typeof res.recall === 'number' ? res.recall.toFixed(3) : '—'}
                                    </td>
                                    <td className="px-4 py-3 text-gray-500 text-xs">
                                        {res.candidate_count != null ? res.candidate_count : '—'}
                                    </td>
                                    <td className="px-4 py-3">
                                        {res.error ? (
                                            <span className="bg-red-100 text-red-700 text-[10px] px-2 py-0.5 rounded-full font-medium">Error</span>
                                        ) : (
                                            <span className="bg-green-100 text-green-700 text-[10px] px-2 py-0.5 rounded-full font-medium">OK</span>
                                        )}
                                    </td>
                                </tr>
                                {expandedRows.has(res.conv_id) && (
                                    <tr className="bg-gray-50/50">
                                        <td colSpan={5} className="px-5 py-4">
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                                                <div>
                                                    <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Pipeline Metrics</p>
                                                    <ul className="space-y-1 text-xs text-gray-600">
                                                        <li>Recall@K: <span className="font-semibold text-blue-600">{typeof res.recall === 'number' ? res.recall.toFixed(3) : '—'}</span></li>
                                                        <li>Candidates: <span className="font-medium">{res.candidate_count ?? '—'}</span></li>
                                                    </ul>
                                                </div>
                                                <div>
                                                    <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Results</p>
                                                    <div className="mb-3">
                                                        <span className="text-[11px] text-gray-400 font-medium">Ground Truth</span>
                                                        <p className="font-mono text-xs bg-white border border-gray-200 rounded-xl p-2.5 mt-1 break-words text-gray-700">
                                                            {Array.isArray(res.ground_truth)
                                                                ? res.ground_truth.join(', ')
                                                                : String(res.ground_truth ?? '—')}
                                                        </p>
                                                    </div>
                                                    <div>
                                                        <span className="text-[11px] text-gray-400 font-medium">Recommendations</span>
                                                        <RecommendationsList recommendations={res.recommendations} />
                                                    </div>
                                                </div>
                                                {res.error && (
                                                    <div className="col-span-2 bg-red-50 text-red-700 p-3 rounded-xl border border-red-200 text-xs">
                                                        Error: {res.error}
                                                    </div>
                                                )}
                                            </div>
                                        </td>
                                    </tr>
                                )}
                            </>
                        ))}
                    </tbody>
                </table>
                {results.length === 0 && (
                    <div className="text-center text-gray-400 py-10 text-sm">No results recorded for this run.</div>
                )}
            </div>
        </div>
    );
}
