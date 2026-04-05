from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.chat.service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    conversation: str


class ChatResponse(BaseModel):
    response: str
    recommendations: list
    agent_trace: list = []
    debug_info: dict = {}


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming endpoint. Returns NDJSON lines where each line is a JSON event:
      - {"event": "node", "node": "profiler"|"retrieval"|"reranking"|"generation",
         "status": "running"|"done", "message": "..."}
      - {"event": "result", "response": "...", "recommendations": [...], "agent_trace": [...]}
    """
    service = ChatService()

    async def generate():
        try:
            async for chunk in service.chat_stream(request.conversation):
                yield chunk
        except Exception as e:
            import json
            yield json.dumps({"event": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},  # needed for nginx not to buffer SSE
    )


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Classic (non-streaming) endpoint — kept for backward compatibility."""
    service = ChatService()
    try:
        result = await service.chat(request.conversation)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
