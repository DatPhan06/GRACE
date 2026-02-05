import { useState } from 'react';
import { EvaluateRequest, runEvaluation } from '@/lib/api';

interface RunEvaluationFormProps {
    onRunComplete?: () => void;
}

export default function RunEvaluationForm({ onRunComplete }: RunEvaluationFormProps) {
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState<EvaluateRequest>({
        dataset: 'redial',
        sample_size: 10,
        start_index: 0,
        n_sample: 100,
        top_k: 10,
        model: 'cohere'
    });
    const [statusMessage, setStatusMessage] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setStatusMessage(null);
        try {
            const result = await runEvaluation(formData);
            setStatusMessage(`Success: ${result.message}`);
            if (onRunComplete) onRunComplete();
        } catch (error) {
            setStatusMessage("Error: Failed to run evaluation");
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: name === 'dataset' || name === 'model' ? value : Number(value)
        }));
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 mb-6">
            <h3 className="text-xl font-semibold mb-4 text-gray-800">Run New Evaluation</h3>
            <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Dataset</label>
                    <select
                        name="dataset"
                        value={formData.dataset}
                        onChange={handleChange}
                        className="w-full p-2 border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        <option value="redial">Redial</option>
                        <option value="inspired">Inspired</option>
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
                    <select
                        name="model"
                        value={formData.model}
                        onChange={handleChange}
                        className="w-full p-2 border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        <option value="cohere">Cohere</option>
                        <option value="llm">LLM</option>
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Sample Size</label>
                    <input
                        type="number"
                        name="sample_size"
                        min="1"
                        max="1000"
                        value={formData.sample_size}
                        onChange={handleChange}
                        className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Top K</label>
                    <input
                        type="number"
                        name="top_k"
                        min="1"
                        max="50"
                        value={formData.top_k}
                        onChange={handleChange}
                        className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">N Sample (Candidates)</label>
                    <input
                        type="number"
                        name="n_sample"
                        min="10"
                        max="600"
                        value={formData.n_sample}
                        onChange={handleChange}
                        className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Start Index</label>
                    <input
                        type="number"
                        name="start_index"
                        min="0"
                        value={formData.start_index}
                        onChange={handleChange}
                        className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                <div className="md:col-span-2 mt-2">
                    <button
                        type="submit"
                        disabled={loading}
                        className={`w-full py-2 px-4 rounded-md text-white font-medium transition-colors ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
                            }`}
                    >
                        {loading ? 'Running Evaluation...' : 'Start Evaluation'}
                    </button>
                    {statusMessage && (
                        <p className={`mt-2 text-sm ${statusMessage.startsWith("Error") ? "text-red-500" : "text-green-600"}`}>
                            {statusMessage}
                        </p>
                    )}
                </div>
            </form>
        </div>
    );
}
