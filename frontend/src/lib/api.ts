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
