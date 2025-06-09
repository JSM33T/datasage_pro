import os
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from api import document,indexing,chat,chat_session

load_dotenv()


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse(Path("static/index.html"))

@app.post("/api/auth/login")
def login(payload: dict = Body(...)):
    if payload.get("password") == os.getenv("AUTH_PASSWORD"):
        return {"token": os.getenv("AUTH_TOKEN")}
    raise HTTPException(status_code=401, detail="Invalid password")

@app.middleware("http")
async def AuthMiddleware(request: Request, call_next):
    if AUTH_REQUIRED and not request.url.path.startswith("/static") and request.url.path != "/":
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header.strip() != AUTH_TOKEN:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing Authorization token"}
            )
    return await call_next(request)


AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "false").lower() == "true"
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if AUTH_REQUIRED and not request.url.path.startswith("/static") and request.url.path != "/":
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header.strip() != AUTH_TOKEN:
            return JSONResponse(
    status_code=401,
    content={"detail": "Invalid or missing Authorization token"}
)
    return await call_next(request)

app.include_router(document.router, prefix="/api/document")
app.include_router(indexing.router, prefix="/api/indexing")
app.include_router(chat.router, prefix="/api/chat")
app.include_router(chat_session.router, prefix="/api/chat_session")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
