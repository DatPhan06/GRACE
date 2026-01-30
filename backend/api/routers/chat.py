from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.chat.service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    conversation: str

class ChatResponse(BaseModel):
    response: str
    recommendations: list
    debug_info: dict

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    service = ChatService()
    try:
        result = await service.chat(request.conversation)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
