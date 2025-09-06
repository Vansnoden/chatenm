from datetime import datetime
from pydantic import BaseModel

    

class ChatSession(BaseModel):
    id: int
    created_at: datetime


class ChatMessageBase(BaseModel):
    session_id: int
    sender_type: str
    content: str


class ChatMessage(ChatMessageBase):
    id: int
    created_at: datetime