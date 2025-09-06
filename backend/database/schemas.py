from datetime import datetime
from pydantic import BaseModel, UUID4


class UserBase(BaseModel):
    username: str
    fullname: str
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: UUID4
    role_id: UUID4 = ''
    is_active: bool

    class Config:
        from_attributes = True


class UserRoleBase(BaseModel):
    name: str


class UserRole(UserRoleBase):
    id: UUID4


class ChatSessionBase(BaseModel):
    user_id: UUID4
    timestamp: datetime


class ChatSession(ChatSessionBase):
    id: UUID4


class MessageBase(BaseModel):
    session_id: UUID4
    sender_type: str
    content: str
    timestamp: datetime


class Message(MessageBase):
    id: UUID4