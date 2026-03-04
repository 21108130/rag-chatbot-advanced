
from __future__ import annotations

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.auth.models import Base, User, APIKey
from src.utils.logger import logger


SECRET_KEY    = secrets.token_urlsafe(32)
ALGORITHM     = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthManager:


    def __init__(self, db_url: str = "sqlite:///./data/users.db") -> None:
        self.engine        = create_engine(db_url, connect_args={"check_same_thread": False})
        self.SessionLocal  = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        logger.info(f"[Auth] Database ready at {db_url}")



    def get_db(self) -> Session:
        return self.SessionLocal()



    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)


    def create_access_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        expire = datetime.utcnow() + (
            expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        payload = {"sub": user_id, "exp": expire, "type": "access"}
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    def decode_token(self, token: str) -> Optional[str]:
        """Return user_id from a valid token, or None."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload.get("sub")
        except JWTError:
            return None



    def create_user(
        self,
        username: str,
        email: str,
        password: str,
    ) -> User:
        db = self.get_db()
        try:

            if db.query(User).filter(User.username == username).first():
                raise ValueError(f"Username '{username}' already taken.")
            if db.query(User).filter(User.email == email).first():
                raise ValueError(f"Email '{email}' already registered.")

            user_id = str(uuid4())
            user = User(
                id               = user_id,
                username         = username,
                email            = email,
                hashed_password  = self.hash_password(password),
                collection_name  = f"user_{user_id.replace('-','')[:16]}",
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
            if not user or not self.verify_password(password, user.hashed_password):
                logger.warning(f"[Auth] Failed login attempt for: {username}")
                return None
            user.last_login = datetime.utcnow()
            db.commit()
            db.refresh(user)
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
            api_key.last_used = datetime.utcnow()
            db.commit()
            return self.get_user_by_id(api_key.user_id)
        finally:
            db.close()



_auth_manager: Optional[AuthManager] = None

def get_auth_manager() -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


