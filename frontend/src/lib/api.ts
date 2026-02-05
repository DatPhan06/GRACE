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
        const response = await axios.get<EvaluationRunResponse[]>(`${API_URL}/evaluate/runs`, {
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
        const response = await axios.get<EvaluationRunResponse>(`${API_URL}/evaluate/runs/${runId}`);
        return response.data;
    } catch (error) {
        console.error("Error fetching run details:", error);
        throw error;
    }
};
