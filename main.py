import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from api import chat_session, chat_session_local, document, chat, indexing, indexing_local

app = FastAPI()

# Static folders
app.mount("/static", StaticFiles(directory="static/browser"), name="static")
app.mount("/resources", StaticFiles(directory="resources"), name="resources")

# Root index.html
@app.get("/")
def serve_root():
    return FileResponse(Path("static/browser/index.html"))

# Login endpoint
@app.post("/api/auth/login")
def login(payload: dict = Body(...)):
    if payload.get("password") == os.getenv("AUTH_PASSWORD"):
        return {"token": os.getenv("AUTH_TOKEN")}
    raise HTTPException(status_code=401, detail="Invalid password")

# Auth middleware for /api/*
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "false").lower() == "true"
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if AUTH_REQUIRED and request.url.path.startswith("/api"):
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header.strip() != AUTH_TOKEN:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing Authorization token"})
    return await call_next(request)

#Router setup based on model
# if os.getenv("MODEL", "openai").lower() == "openai":
#     app.include_router(indexing.router, prefix="/api/indexing")
#     app.include_router(chat_session.router, prefix="/api/chat_session")
# else:
#     app.include_router(indexing_local.router, prefix="/api/indexing")
#     app.include_router(chat_session_local.router, prefix="/api/chat_session")


app.include_router(indexing.router, prefix="/api/indexing")
app.include_router(chat_session.router, prefix="/api/chat_session")

app.include_router(document.router, prefix="/api/document")
app.include_router(chat.router, prefix="/api/chat")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/{full_path:path}")
def catch_all(full_path: str):
    target = Path("static/browser") / full_path
    if target.exists() and target.is_file():
        return FileResponse(target)
    return FileResponse(Path("static/browser/index.html"))