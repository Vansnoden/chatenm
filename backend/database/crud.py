# -*- coding: utf-8 -*-

import csv
from datetime import datetime
import json
import os
import shutil
from sqlalchemy import and_
from sqlalchemy.orm import Session
from tqdm import tqdm

from database.utils import get_uuid
from . import models, schemas
from passlib.context import CryptContext
from pathlib import Path
import logging, traceback


logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)


def get_chat_sessions(db: Session):
    return db.query(models.ChatSession).all()


def get_chat_session(db: Session, chat_session_id: int):
    return db.query(models.ChatSession)\
        .filter(models.ChatSession.id == chat_session_id).first()


def get_chat_session_messages(db: Session, chat_session_id: int):
    return db.query(models.ChatMessage)\
        .filter(models.ChatMessage.session_id == chat_session_id).all()


def create_chat_session(db: Session):
    db_chat_session = models.ChatSession()
    db.add(db_chat_session)
    db.commit()
    db.refresh(db_chat_session)
    return db_chat_session


def delete_chat_session(db: Session, chat_session_id: int):
    res = db.query(models.ChatSession)\
        .filter(models.ChatSession.id == chat_session_id).delete()
    db.commit()
    return res


def create_chat_message(db: Session, chat_session_id: int, sender_type: str, msg: str):
    db_chat_msg = models.ChatMessage(
        session_id = chat_session_id,
        sender_type =  sender_type,
        content = msg
    )
    db.add(db_chat_msg)
    db.commit()
    db.refresh(db_chat_msg)
    return db_chat_msg


def delete_chat_message(db: Session, msg_id: int):
    res = db.query(models.ChatMessage)\
        .filter(models.ChatMessage.id == msg_id).delete()
    db.commit()
    return res


def ask_question_to_llm(db: Session,  session_id: int, msg:str):
    answer = "bot answer"
    create_chat_message(db, session_id, 'bot', answer)
    return answer