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


class UserRole(Base):
    __tablename__ = "user_roles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role_id = Column(UUID(as_uuid=True), ForeignKey("user_roles.id"), nullable=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    fullname = Column(String)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True) 
    create_date = Column(DateTime, default=datetime.datetime.now(), nullable=False)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.now(), nullable=False)


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    sender_type = Column(String, nullable=False) # (user/bot)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.now(), nullable=False)





    



