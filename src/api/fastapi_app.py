
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import settings
from src.auth.auth_manager import get_auth_manager
from src.auth.models import User
from src.observability.metrics import get_metrics
from src.retrieval.advanced_rag_pipeline import AdvancedRAGPipeline
from src.utils.logger import logger

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    limiter = None
    RATE_LIMIT_AVAILABLE = False
    logger.warning("[API] slowapi not installed — rate limiting disabled.")


app = FastAPI(
    title       = "RAG Chatbot API",
    description = "Advanced RAG pipeline with hybrid search, reranking, and streaming.",
    version     = "2.0.0",
    docs_url    = "/docs",
)

if RATE_LIMIT_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Global pipeline instance ───────────────────────────────────────────────────
_pipeline: Optional[AdvancedRAGPipeline] = None

def get_pipeline() -> AdvancedRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AdvancedRAGPipeline()
    return _pipeline




async def get_current_user(
    authorization: Optional[str] = Header(None),
    x_api_key:     Optional[str] = Header(None),
) -> User:
    auth = get_auth_manager()


    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        user  = auth.get_current_user(token)
        if user:
            return user


    if x_api_key:
        user = auth.validate_api_key(x_api_key)
        if user:
            return user

    raise HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail      = "Invalid or missing authentication credentials.",
        headers     = {"WWW-Authenticate": "Bearer"},
    )




class RegisterRequest(BaseModel):
    username: str
    email:    str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    username:     str

class ChatRequest(BaseModel):
    query:           str
    conversation_id: Optional[str] = None
    top_k:           Optional[int] = None
    stream:          bool = False

class APIKeyRequest(BaseModel):
    name: str

class APIKeyResponse(BaseModel):
    api_key: str
    name:    str
    message: str




@app.post("/auth/register", tags=["Authentication"])
async def register(request: RegisterRequest):
    """Register a new user account."""
    auth = get_auth_manager()
    try:
        user = auth.create_user(request.username, request.email, request.password)
        return {"message": "User created successfully.", "user_id": user.id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(request: LoginRequest):
    """Login and receive a JWT access token."""
    auth  = get_auth_manager()
    user  = auth.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = auth.create_access_token(user.id)
    return TokenResponse(
        access_token = token,
        user_id      = user.id,
        username     = user.username,
    )


@app.post("/auth/api-keys", response_model=APIKeyResponse, tags=["Authentication"])
async def create_api_key(
    request: APIKeyRequest,
    user:    User = Depends(get_current_user),
):
    """Generate a new API key for programmatic access."""
    auth = get_auth_manager()
    raw_key = auth.create_api_key(user.id, request.name)
    return APIKeyResponse(
        api_key = raw_key,
        name    = request.name,
        message = "Store this key safely — it will not be shown again.",
    )


@app.post("/documents/upload", tags=["Documents"])
async def upload_document(
    file:     UploadFile = File(...),
    user:     User = Depends(get_current_user),
    pipeline: AdvancedRAGPipeline = Depends(get_pipeline),
):
    """Upload and index a document into the user's knowledge base."""

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {ext}")


    dest = settings.upload_path / f"{uuid4()}_{file.filename}"
    content = await file.read()

    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_file_size_mb}MB limit.")

    dest.write_bytes(content)

    result = pipeline.ingest_document(str(dest), user_id=user.id)
    return result.model_dump()


@app.get("/documents", tags=["Documents"])
async def list_documents(
    user:     User = Depends(get_current_user),
    pipeline: AdvancedRAGPipeline = Depends(get_pipeline),
):
    """List all documents in the user's knowledge base."""
    stats = pipeline.get_kb_stats(user_id=user.id)
    return stats



@app.post("/chat", tags=["Chat"])
async def chat(
    request:  ChatRequest,
    user:     User = Depends(get_current_user),
    pipeline: AdvancedRAGPipeline = Depends(get_pipeline),
):

    from src.utils.models import ChatRequest as PipelineChatRequest

    if request.stream:

        def generate_stream():
            for token in pipeline.chat_stream(
                query           = request.query,
                user_id         = user.id,
                conversation_id = request.conversation_id,
                top_k           = request.top_k,
            ):
                yield token

        return StreamingResponse(generate_stream(), media_type="text/plain")


    pipeline_request = PipelineChatRequest(
        query           = request.query,
        conversation_id = request.conversation_id,
        top_k           = request.top_k,
    )
    response = pipeline.chat(pipeline_request, user_id=user.id)
    return response.model_dump()




@app.get("/health", tags=["Operations"])
async def health_check():
    """Service health check."""
    return {
        "status":    "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version":   "2.0.0",
    }


@app.get("/metrics/summary", tags=["Operations"])
async def metrics_summary(user: User = Depends(get_current_user)):
    """Get system performance metrics."""
    metrics = get_metrics()
    return metrics.get_summary()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
