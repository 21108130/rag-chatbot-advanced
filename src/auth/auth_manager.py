from __future__ import annotations

import os
import secrets
import hashlib
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from jose import JWTError, jwt
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session

from src.auth.models import Base, User, APIKey
from src.utils.logger import logger


SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

DB_PATH = "/tmp/rag_users.db"
DB_URL  = f"sqlite:///{DB_PATH}"


def _ensure_fresh_db(engine) -> None:
    """Drop and recreate tables if any expected column is missing."""
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        needs_reset = False
        if "users" not in existing_tables:
            needs_reset = True
        else:
            existing_cols = {c["name"] for c in inspector.get_columns("users")}
            expected_cols = {"id", "username", "email", "hashed_password",
                             "is_active", "created_at", "last_login", "collection_name"}
            if not expected_cols.issubset(existing_cols):
                logger.warning(f"[Auth] Schema mismatch — missing cols: {expected_cols - existing_cols}. Resetting DB.")
                needs_reset = True

        if needs_reset:
            Base.metadata.drop_all(engine)
            Base.metadata.create_all(engine)
            logger.info("[Auth] DB schema recreated fresh.")
        else:
            logger.info("[Auth] DB schema OK.")
    except Exception as e:
        logger.warning(f"[Auth] Schema check failed, recreating DB: {e}")
        try:
            Base.metadata.drop_all(engine)
        except Exception:
            pass
        Base.metadata.create_all(engine)


class AuthManager:

    def __init__(self, db_url: str = DB_URL) -> None:
        # Always redirect any ./data/... path to /tmp on Streamlit Cloud
        if "data/users.db" in db_url or db_url == "sqlite:///./data/users.db":
            db_url = DB_URL
        elif db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////tmp"):
            db_path = db_url.replace("sqlite:///./", "").replace("sqlite:///", "")
            db_dir  = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(db_dir, exist_ok=True)

        self.engine       = create_engine(db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(bind=self.engine)
        _ensure_fresh_db(self.engine)
        logger.info(f"[Auth] Database ready at {db_url}")

    def get_db(self) -> Session:
        return self.SessionLocal()

    
    def hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def verify_password(self, plain: str, hashed: str) -> bool:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

 

    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        expire  = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        payload = {"sub": user_id, "exp": expire, "type": "access"}
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    def decode_token(self, token: str) -> Optional[str]:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload.get("sub")
        except JWTError:
            return None

   

    def create_user(self, username: str, email: str, password: str) -> User:
        db = self.get_db()
        try:
            if db.query(User).filter(User.username == username).first():
                raise ValueError(f"Username '{username}' already taken.")
            if db.query(User).filter(User.email == email).first():
                raise ValueError(f"Email '{email}' already registered.")
            user_id = str(uuid4())
            user = User(
                id              = user_id,
                username        = username,
                email           = email,
                hashed_password = self.hash_password(password),
                collection_name = f"user_{user_id.replace('-', '')[:16]}",
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"[Auth] Created user: {username}")
            return user
        finally:
            db.close()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        db = self.get_db()
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                logger.warning(f"[Auth] Unknown user: {username}")
                return None
            if not self.verify_password(password, user.hashed_password):
                logger.warning(f"[Auth] Wrong password for: {username}")
                return None
            logger.info(f"[Auth] Successful login: {username}")
            return user
        finally:
            db.close()

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        db = self.get_db()
        try:
            return db.query(User).filter(User.id == user_id).first()
        finally:
            db.close()

    def get_current_user(self, token: str) -> Optional[User]:
        user_id = self.decode_token(token)
        if not user_id:
            return None
        return self.get_user_by_id(user_id)



    def create_api_key(self, user_id: str, name: str) -> str:
        raw_key  = f"rag_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        db = self.get_db()
        try:
            api_key = APIKey(
                id       = str(uuid4()),
                user_id  = user_id,
                key_hash = key_hash,
                name     = name,
            )
            db.add(api_key)
            db.commit()
            logger.info(f"[Auth] API key created for user {user_id}: {name}")
            return raw_key
        finally:
            db.close()

    def validate_api_key(self, raw_key: str) -> Optional[User]:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        db = self.get_db()
        try:
            api_key = db.query(APIKey).filter(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True,
            ).first()
            if not api_key:
                return None
            try:
                api_key.last_used = datetime.utcnow()
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning(f"[Auth] Could not update last_used: {e}")
            return self.get_user_by_id(api_key.user_id)
        finally:
            db.close()




_auth_manager: Optional[AuthManager] = None

def get_auth_manager() -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
