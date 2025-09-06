# -*- coding: utf-8 -*-

import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from database import crud
from database.database import SessionLocal
from sqlalchemy.orm import Session
import logging, sys

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)
if sys.version_info[0] >= 3:
    unicode = str


SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 3600 * 24
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ITEMS_PER_PAGE = 100

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5173/*",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5173/*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MSG_TEMPLATE = """
    For the following instructions, use tools outputs to get full file paths
    save the worldclim data in ./data/environmental
    save elevation data in ./data/environmental
    save GBIF data in ./data/gbif_data/
    save final results in ./data/outputs

    {prompt}

"""

@app.post("/sessions")
async def get_sessions(db: Session = Depends(get_db)):
    chat_sessions = crud.get_chat_sessions(db)
    return chat_sessions


@app.post("/user/session")
async def get_session(
    session_id: int, 
    db: Session = Depends(get_db)):
    if session_id:
        chat_session_messages = crud.get_chat_session_messages(db, session_id)
        return chat_session_messages
    else:
        raise HTTPException(status_code=403, detail="Unauthorized action")


class MessageRequest(BaseModel):
    msg: str
    session_id: int = 0


@app.post("/user/send")
async def send_message(
    req: MessageRequest ,
    db: Session = Depends(get_db)):
    msg = req.msg
    session_id = req.session_id
    if msg:
        if session_id:
            crud.create_chat_message(db, session_id, 'user', msg)
            response = crud.ask_question_to_llm(db, session_id, MSG_TEMPLATE.format(prompt = msg))
        else:
            new_session = crud.create_chat_session(db)
            crud.create_chat_message(db, new_session.id, 'user', msg)
            response = crud.ask_question_to_llm(db, new_session.id, MSG_TEMPLATE.format(prompt = msg))
        return response
    else:
        raise HTTPException(status_code=403, detail="Unauthorized action")
    

@app.get("/file/{filename}")
async def get_file(filename: str):
    file_path = f"uploads/{filename}"
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)
