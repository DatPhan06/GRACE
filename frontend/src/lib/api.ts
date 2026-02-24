import axios from 'axios';

const API_URL = 'http://localhost:8000';

export interface MovieRecommendation {
    movieId: string;
    title: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    year?: any;
    plot?: string;
    poster?: string;
    score?: number;
}

export interface ChatResponse {
    response: string;
    recommendations: MovieRecommendation[];
    debug_info?: Record<string, unknown>;
}

export const sendMessage = async (conversation: string): Promise<ChatResponse> => {
    try {
        const response = await axios.post<ChatResponse>(`${API_URL}/chat/`, {
            conversation
        });
        return response.data;
    } catch (error) {
        console.error("Error sending message:", error);
        throw error;
    }
};

// Evaluation Interfaces

export interface EvaluateRequest {
    dataset: "inspired" | "redial";
    sample_size: number;
    start_index: number;
    n_sample: number;
    top_k: number;
    model: "llm" | "cohere";
}

export interface EvaluationResultResponse {
    conv_id: string;
    recall: number;
    ground_truth: string[] | unknown;
    recommendations: string[] | unknown;
    candidate_count?: number;

    recall_retrieval?: number;
    recall_semantic?: number;
    recall_content?: number;
    recall_collab?: number;

    semantic_count?: number;
    content_count?: number;
    collab_count?: number;

    error?: string;
}

export interface EvaluationRunResponse {
    id: number;
    dataset: string;
    sample_size: number;
    n_sample: number;
    top_k: number;
    model: string;
    avg_recall: number;

    avg_recall_retrieval?: number;
    avg_recall_semantic?: number;
    avg_recall_content?: number;
    avg_recall_collab?: number;

    status?: string;
    timestamp?: string;
    results?: EvaluationResultResponse[];
}

export interface EvaluateResponse extends EvaluationRunResponse {
    output_dir: string;
    message: string;
}

// Evaluation API Functions

export const runEvaluation = async (params: EvaluateRequest): Promise<EvaluateResponse> => {
    try {
        const response = await axios.post<EvaluateResponse>(`${API_URL}/evaluate/`, params);
        return response.data;
    } catch (error) {
        console.error("Error running evaluation:", error);
        throw error;
    }
};

export const getEvaluationRuns = async (skip = 0, limit = 100): Promise<EvaluationRunResponse[]> => {
    try {
        const response = await axios.get<EvaluationRunResponse[]>(`${API_URL}/tracing/runs`, {
            params: { skip, limit }
        });
        return response.data;
    } catch (error) {
        console.error("Error fetching run history:", error);
        throw error;
    }
};

export const getEvaluationRun = async (runId: number): Promise<EvaluationRunResponse> => {
    try {
        const response = await axios.get<EvaluationRunResponse>(`${API_URL}/tracing/runs/${runId}`);
        return response.data;
    } catch (error) {
        console.error("Error fetching run details:", error);
        throw error;
    }
};

// Step-based Evaluation APIs

export interface InitRunResponse {
    run_id: number;
    message: string;
    status: string;
}

export interface StepResponse {
    run_id: number;
    batch_id?: number;
    message: string;
    status: string;
    count?: number;
}

export interface BatchStepExecutionResponse {
    id: number;
    run_id: number;
    step_type: string;
    version: number;
    config: Record<string, unknown>;
    status: string;
    created_at: string;
}

export const initializeRun = async (dataset: "inspired" | "redial", sample_size: number, start_index: number = 0): Promise<InitRunResponse> => {
    const response = await axios.post<InitRunResponse>(`${API_URL}/evaluate/init`, { dataset, sample_size, start_index });
    return response.data;
};

export const runSummarizationStep = async (run_id: number): Promise<StepResponse> => {
    const response = await axios.post<StepResponse>(`${API_URL}/evaluate/step/summarize`, { run_id });
    return response.data;
};

export const runRetrievalStep = async (run_id: number, n_sample: number, input_batch_id?: number): Promise<StepResponse> => {
    const response = await axios.post<StepResponse>(`${API_URL}/evaluate/step/retrieve`, { run_id, n_sample, input_batch_id });
    return response.data;
};

export const runRerankingStep = async (run_id: number, top_k: number, model: "llm" | "cohere", input_batch_id?: number): Promise<StepResponse> => {
    const response = await axios.post<StepResponse>(`${API_URL}/evaluate/step/rerank`, { run_id, top_k, model, input_batch_id });
    return response.data;
};

export const getStepHistory = async (run_id: number): Promise<BatchStepExecutionResponse[]> => {
    const response = await axios.get<BatchStepExecutionResponse[]>(`${API_URL}/tracing/runs/${run_id}/history`);
    return response.data;
};

export const getStepsByType = async (stepType?: string): Promise<BatchStepExecutionResponse[]> => {
    const response = await axios.get<BatchStepExecutionResponse[]>(`${API_URL}/tracing/steps`, {
        params: stepType ? { step_type: stepType } : {}
    });
    return response.data;
};

// Batch Detail APIs

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface BatchDetailItem {
    conv_id: string;
    // Summarization
    user_preferences?: string;
    // Retrieval
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    candidates?: any[];
    candidate_count?: number;
    semantic_count?: number;
    content_count?: number;
    collab_count?: number;
    // Reranking
    model_used?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    reranked_candidates?: any[];
    reranked_count?: number;
    recall?: number;
}

export interface BatchDetailResponse {
    id: number;
    run_id: number;
    step_type: string;
    version: number;
    config: Record<string, unknown>;
    status: string;
    created_at: string;
    items: BatchDetailItem[];
}

export const getStepBatchDetail = async (batchId: number): Promise<BatchDetailResponse> => {
    const response = await axios.get<BatchDetailResponse>(`${API_URL}/tracing/batches/${batchId}/details`);
    return response.data;
};
