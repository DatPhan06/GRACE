import { EvaluationRunResponse, EvaluationResultResponse } from '@/lib/api';
import { useState } from 'react';

interface RunDetailViewProps {
    run: EvaluationRunResponse;
}

export default function RunDetailView({ run }: RunDetailViewProps) {
    const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

    const toggleRow = (convId: string) => {
        const newExpanded = new Set(expandedRows);
        if (newExpanded.has(convId)) {
            newExpanded.delete(convId);
        } else {
            newExpanded.add(convId);
        }
        setExpandedRows(newExpanded);
    };

    const results = run.results || [];

    return (
        <div className="mt-4">
            <div className="p-4 bg-blue-50/50 rounded-lg border border-blue-100 mb-6">
                {/* <h3 className="text-lg font-semibold text-gray-800">Run Details #{run.id}</h3> */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="bg-white p-3 rounded border">
                        <span className="text-gray-500 block">Avg Recall (Final)</span>
                        <span className="font-bold text-lg text-blue-600">{run.avg_recall?.toFixed(3)}</span>
                    </div>
                    <div className="bg-white p-3 rounded border">
                        <span className="text-gray-500 block">Retrieval Recall</span>
                        <span className="font-bold text-lg text-gray-700">{run.avg_recall_retrieval?.toFixed(3) || "0.000"}</span>
                    </div>
                    <div className="bg-white p-3 rounded border">
                        <span className="text-gray-500 block">Semantic Recall</span>
                        <span className="font-bold text-lg text-gray-700">{run.avg_recall_semantic?.toFixed(3) || "0.000"}</span>
                    </div>
                    <div className="bg-white p-3 rounded border">
                        <span className="font-bold text-lg text-gray-700">{run.avg_recall_content?.toFixed(3) || "0.000"}</span>
                    </div>
                </div>
            </div>

            <div className="overflow-x-auto rounded-lg border border-gray-200">
                <table className="w-full text-sm text-left">
                    <thead className="bg-gray-50 text-gray-700 uppercase text-xs">
                        <tr>
                            <th className="px-6 py-3 w-4"></th>
                            <th className="px-6 py-3">Conv ID</th>
                            <th className="px-6 py-3">Recall</th>
                            <th className="px-6 py-3">Retrieval (N)</th>
                            <th className="px-6 py-3">Sources (Sem/Con/Col)</th>
                            <th className="px-6 py-3">Status</th>
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
                                    <td className="px-6 py-4 text-gray-400">
                                        {expandedRows.has(res.conv_id) ? '▼' : '▶'}
                                    </td>
                                    <td className="px-6 py-4 font-medium">{res.conv_id}</td>
                                    <td className="px-6 py-4 font-semibold text-blue-600">
                                        {typeof res.recall === 'number' ? res.recall.toFixed(3) : res.recall}
                                    </td>
                                    <td className="px-6 py-4 text-gray-600">
                                        {typeof res.recall_retrieval === 'number' ? res.recall_retrieval.toFixed(3) : '-'}
                                    </td>
                                    <td className="px-6 py-4 text-xs text-gray-500">
                                        {typeof res.recall_semantic === 'number' ? res.recall_semantic.toFixed(3) : '-'} /
                                        {typeof res.recall_content === 'number' ? res.recall_content.toFixed(3) : '-'} /
                                        {typeof res.recall_collab === 'number' ? res.recall_collab.toFixed(3) : '-'}
                                    </td>
                                    <td className="px-6 py-4">
                                        {res.error ? (
                                            <span className="text-red-500 text-xs font-bold">ERROR</span>
                                        ) : (
                                            <span className="text-green-500 text-xs font-bold">OK</span>
                                        )}
                                    </td>
                                </tr>
                                {expandedRows.has(res.conv_id) && (
                                    <tr className="bg-gray-50">
                                        <td colSpan={6} className="px-6 py-4">
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                                                <div>
                                                    <h4 className="font-semibold text-gray-700 mb-2">Metrics Detail</h4>
                                                    <ul className="space-y-1 text-gray-600">
                                                        <li>Candidate Count: {res.candidate_count}</li>
                                                        <li>Semantic Candidates: {res.semantic_count}</li>
                                                        <li>Content Candidates: {res.content_count}</li>
                                                        <li>Collaborative Candidates: {res.collab_count}</li>
                                                    </ul>
                                                </div>
                                                <div>
                                                    <h4 className="font-semibold text-gray-700 mb-2">Data</h4>
                                                    <div className="mb-2">
                                                        <span className="font-medium text-xs uppercase text-gray-500">Ground Truth:</span>
                                                        <p className="font-mono text-xs bg-white p-2 border rounded mt-1">
                                                            {Array.isArray(res.ground_truth) ? res.ground_truth.join(", ") : String(res.ground_truth)}
                                                        </p>
                                                    </div>
                                                    <div>
                                                        <span className="font-medium text-xs uppercase text-gray-500">Recommendations:</span>
                                                        <p className="font-mono text-xs bg-white p-2 border rounded mt-1">
                                                            {Array.isArray(res.recommendations) ? res.recommendations.join(", ") : String(res.recommendations)}
                                                        </p>
                                                    </div>
                                                </div>
                                                {res.error && (
                                                    <div className="col-span-2 bg-red-50 text-red-700 p-3 rounded border border-red-200">
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
            </div>
        </div>
    );
}
