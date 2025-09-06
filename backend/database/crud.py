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



DELIMITER = ","


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 0):
    if skip and limit:
        return db.query(models.User).offset(skip).limit(limit).all()
    else:
        return db.query(models.User).all()


def create_user(db: Session, user: schemas.UserCreate):
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    db_user = models.User(
        username=user.username,
        email=user.email, 
        fullname=user.fullname,
        is_active=True,
        hashed_password=pwd_context.hash(user.password))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def delete_user(db: Session, user_id: int):
    res = db.query(models.User).filter(models.User.id == user_id).delete()
    db.commit()
    return res


def create_user_role(db: Session, role: schemas.UserRoleBase):
    db_role = models.UserRole(
        name=role.name
    )
    db.add(db_role)
    db.commit()
    db.refresh(db_role)
    return db_role

def get_user_role_by_name(db: Session, name: str):
    return db.query(models.UserRole).filter(models.UserRole.name == name).first()


def get_system_roles(db: Session, skip: int = 0, limit: int = 0):
    if skip and limit:
        return db.query(models.UserRole).offset(skip).limit(limit).all()
    else:
        return db.query(models.UserRole).all()
    

def assign_user_role(db: Session, user_id: int, role_id: int):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    db_role = db.query(models.UserRole).filter(models.UserRole.id == role_id).first()
    if db_role and db_user:
        db_user.role_id = role_id
        db.commit()
    return db_user


def delete_user_role(db: Session, role_id: int):
    res = db.query(models.UserRole).filter(models.UserRole.id == role_id).delete()
    db.commit()
    return res



def ask_question_to_llm(db: Session,  user_id: int, msg:str):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if msg and db_user:
        # do something here
        db.commit()
    return db_user