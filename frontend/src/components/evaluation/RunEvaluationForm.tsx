import { useState, useEffect, useMemo } from 'react';
import { EvaluateRequest, EvaluationRunResponse, runEvaluation, getEvaluationInfo } from '@/lib/api';

interface RunEvaluationFormProps {
    onRunComplete?: () => void;
    existingRuns?: EvaluationRunResponse[];
}

export default function RunEvaluationForm({ onRunComplete, existingRuns = [] }: RunEvaluationFormProps) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [datasetSizes, setDatasetSizes] = useState<Record<string, number>>({});
    const [llmModel, setLlmModel] = useState('');
    const [formData, setFormData] = useState<EvaluateRequest>({
        name: '',
        dataset: 'redial',
        sample_percent: 20,
        n_sample: 400,
        top_k: 10,
        model: 'llm',
    });

    useEffect(() => {
        getEvaluationInfo()
            .then(info => {
                setDatasetSizes(info.dataset_sizes ?? {});
                setLlmModel(info.llm_model ?? '');
            })
            .catch(() => {});
    }, []);

    // Compute the suggested default name based on current config
    const suggestedName = useMemo(() => {
        if (!llmModel) return '';
        const dataset = formData.dataset.toUpperCase();
        const topK = formData.top_k;
        const modelLabel = formData.model === 'llm' ? llmModel : formData.model;
        // Version = how many runs share the same dataset + top_k + model, + 1
        const similar = existingRuns.filter(
            r => r.dataset?.toLowerCase() === formData.dataset &&
                 r.top_k === formData.top_k &&
                 r.model === formData.model
        );
        const version = (similar.length + 1).toFixed(1);
        return `ARGOS - ${dataset} - recall@${topK} [${modelLabel}] - ${version}`;
    }, [llmModel, formData.dataset, formData.top_k, formData.model, existingRuns]);

    const totalSize = datasetSizes[formData.dataset] ?? 0;
    const estimatedCount = totalSize ? Math.max(1, Math.round(totalSize * formData.sample_percent / 100)) : null;

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        try {
            const name = formData.name?.trim() || suggestedName || undefined;
            await runEvaluation({ ...formData, name });
            if (onRunComplete) onRunComplete();
        } catch {
            setError('Failed to start evaluation. Check backend logs.');
            setLoading(false);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: name === 'dataset' || name === 'model' || name === 'name' ? value : Number(value),
        }));
    };

    return (
        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Run Name */}
            <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                    Run Name <span className="text-gray-400">(optional)</span>
                </label>
                <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    placeholder={suggestedName || 'Auto-generated from config'}
                    className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder:text-gray-400"
                />
                {suggestedName && !formData.name && (
                    <p className="text-xs text-gray-400 mt-1">
                        Default: <span className="font-mono">{suggestedName}</span>
                    </p>
                )}
            </div>

            {/* Dataset */}
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Dataset</label>
                <select
                    name="dataset"
                    value={formData.dataset}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    <option value="redial">
                        Redial{datasetSizes['redial'] ? ` (${datasetSizes['redial'].toLocaleString()})` : ''}
                    </option>
                    <option value="inspired">
                        Inspired{datasetSizes['inspired'] ? ` (${datasetSizes['inspired'].toLocaleString()})` : ''}
                    </option>
                </select>
            </div>

            {/* Model */}
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
                <select
                    name="model"
                    value={formData.model}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    <option value="llm">LLM {llmModel ? `(${llmModel})` : ''}</option>
                    <option value="cohere">Cohere</option>
                </select>
            </div>

            {/* Sample % */}
            <div className="md:col-span-2">
                <div className="flex items-center justify-between mb-1">
                    <label className="block text-sm font-medium text-gray-700">Sample Size</label>
                    <span className="text-sm font-semibold text-blue-600">
                        {formData.sample_percent}%
                        {estimatedCount ? (
                            <span className="text-gray-400 font-normal ml-1">
                                (~{estimatedCount.toLocaleString()} conversations)
                            </span>
                        ) : null}
                    </span>
                </div>
                <input
                    type="range"
                    name="sample_percent"
                    min="1"
                    max="100"
                    step="1"
                    value={formData.sample_percent}
                    onChange={handleChange}
                    className="w-full accent-blue-600"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                    <span>1%</span>
                    <span>25%</span>
                    <span>50%</span>
                    <span>75%</span>
                    <span>100%</span>
                </div>
            </div>

            {/* Top K */}
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Top K</label>
                <select
                    name="top_k"
                    value={formData.top_k}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    {[1, 5, 10, 50].map(k => (
                        <option key={k} value={k}>{k}</option>
                    ))}
                </select>
            </div>

            {/* N Candidates */}
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">N Candidates</label>
                <select
                    name="n_sample"
                    value={formData.n_sample}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    {[100, 200, 300, 400, 500, 600].map(n => (
                        <option key={n} value={n}>{n}</option>
                    ))}
                </select>
            </div>

            {/* Submit */}
            <div className="md:col-span-2 mt-2">
                <button
                    type="submit"
                    disabled={loading}
                    className={`w-full py-2 px-4 rounded-md text-white font-medium transition-colors ${
                        loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
                    }`}
                >
                    {loading ? 'Starting...' : 'Start Evaluation'}
                </button>
                {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
            </div>
        </form>
    );
}
