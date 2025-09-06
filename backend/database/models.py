# -*- coding: utf-8 -*-

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, Date, Float, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from .database import Base
import datetime
from sqlalchemy.dialects.postgresql import UUID
import uuid


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.now(), nullable=False)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    sender_type = Column(String, nullable=False) # (user/bot)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.now(), nullable=False)





    



