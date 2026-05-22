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
        <div className="mt-1">
            <ol className="font-mono text-xs bg-white p-2 border rounded space-y-0.5">
                {shown.map((title, i) => (
                    <li key={i} className="flex gap-1.5">
                        <span className="text-gray-400 w-5 shrink-0 text-right">{i + 1}.</span>
                        <span className="break-words">{title}</span>
                    </li>
                ))}
            </ol>
            {hidden > 0 && (
                <button
                    type="button"
                    onClick={() => setExpanded(v => !v)}
                    className="text-xs text-blue-500 hover:text-blue-700 mt-1"
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
        <div className="mt-4">
            {/* Export button */}
            <div className="flex justify-end mb-3">
                <button
                    onClick={handleExport}
                    disabled={exporting || results.length === 0}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                        exporting || results.length === 0
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                            : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
                    }`}
                >
                    <Download size={15} />
                    {exporting ? 'Exporting…' : 'Export TSV'}
                </button>
            </div>

            {/* Run config + primary metrics */}
            <div className="p-4 bg-blue-50/50 rounded-lg border border-blue-100 mb-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-4">
                    <div className="bg-white p-3 rounded border">
                        <span className="text-gray-500 block text-xs">Recall@K (Final)</span>
                        <span className="font-bold text-lg text-blue-600">{run.avg_recall?.toFixed(3) ?? '—'}</span>
                    </div>
                    <div className="bg-white p-3 rounded border">
                        <span className="text-gray-500 block text-xs">Model</span>
                        <span className="font-bold text-base text-gray-700 uppercase">{run.model ?? '—'}</span>
                    </div>
                    <div className="bg-white p-3 rounded border">
                        <span className="text-gray-500 block text-xs">Config</span>
                        <span className="font-bold text-sm text-gray-700">
                            Top-{run.top_k} / N={run.n_sample}
                        </span>
                    </div>
                </div>

            </div>

            {/* Per-conversation table */}
            <div className="overflow-x-auto rounded-lg border border-gray-200">
                <table className="w-full text-sm text-left">
                    <thead className="bg-gray-50 text-gray-700 uppercase text-xs">
                        <tr>
                            <th className="px-4 py-3 w-4"></th>
                            <th className="px-4 py-3">Conv ID</th>
                            <th className="px-4 py-3">Recall@K</th>
                            <th className="px-4 py-3">Candidates</th>
                            <th className="px-4 py-3">Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                        {results.map((res: EvaluationResultResponse) => (
                            <>
                                <tr
                                    key={res.conv_id}
                                    onClick={() => toggleRow(res.conv_id)}
                                    className="hover:bg-gray-50 cursor-pointer transition-colors"
                                >
                                    <td className="px-4 py-3 text-gray-400 text-xs">
                                        {expandedRows.has(res.conv_id) ? '▼' : '▶'}
                                    </td>
                                    <td className="px-4 py-3 font-medium text-gray-800 text-xs max-w-[180px] truncate">
                                        {res.conv_id}
                                    </td>
                                    <td className="px-4 py-3 font-semibold text-blue-600">
                                        {typeof res.recall === 'number' ? res.recall.toFixed(3) : '—'}
                                    </td>
                                    <td className="px-4 py-3 text-gray-500 text-xs">
                                        {res.candidate_count != null ? res.candidate_count : '—'}
                                    </td>
                                    <td className="px-4 py-3">
                                        {res.error ? (
                                            <span className="text-red-500 text-xs font-bold">ERROR</span>
                                        ) : (
                                            <span className="text-green-500 text-xs font-bold">OK</span>
                                        )}
                                    </td>
                                </tr>
                                {expandedRows.has(res.conv_id) && (
                                    <tr className="bg-gray-50">
                                        <td colSpan={5} className="px-6 py-4">
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                                                {/* Left: pipeline metrics */}
                                                <div>
                                                    <h4 className="font-semibold text-gray-700 mb-2 text-xs uppercase tracking-wide">
                                                        Pipeline Metrics
                                                    </h4>
                                                    <ul className="space-y-1 text-gray-600 text-xs">
                                                        <li>Recall@K (Final): <span className="font-medium text-blue-600">{typeof res.recall === 'number' ? res.recall.toFixed(3) : '—'}</span></li>
                                                        <li>Candidates (into Critic): <span className="font-medium">{res.candidate_count ?? '—'}</span></li>
                                                    </ul>
                                                </div>

                                                {/* Right: ground truth + recommendations */}
                                                <div>
                                                    <h4 className="font-semibold text-gray-700 mb-2 text-xs uppercase tracking-wide">
                                                        Results
                                                    </h4>
                                                    <div className="mb-3">
                                                        <span className="font-medium text-xs text-gray-500 uppercase">Ground Truth</span>
                                                        <p className="font-mono text-xs bg-white p-2 border rounded mt-1 break-words">
                                                            {Array.isArray(res.ground_truth)
                                                                ? res.ground_truth.join(', ')
                                                                : String(res.ground_truth ?? '—')}
                                                        </p>
                                                    </div>
                                                    <div>
                                                        <span className="font-medium text-xs text-gray-500 uppercase">Recommendations</span>
                                                        <RecommendationsList recommendations={res.recommendations} />
                                                    </div>
                                                </div>

                                                {res.error && (
                                                    <div className="col-span-2 bg-red-50 text-red-700 p-3 rounded border border-red-200 text-xs">
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
                    <div className="text-center text-gray-400 py-8 text-sm">No results recorded for this run.</div>
                )}
            </div>
        </div>
    );
}
