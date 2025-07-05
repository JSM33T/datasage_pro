import os
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from api import chat_session, document, chat, indexing

app = FastAPI()

# ===== Static folders =====
app.mount("/static", StaticFiles(directory="static/browser"), name="static")
app.mount("/resources", StaticFiles(directory="resources"), name="resources")

# ===== Root index.html =====
@app.get("/")
def serve_root():
    return FileResponse(Path("static/browser/index.html"))

# ===== Login endpoint with USER and ADMIN =====
@app.post("/api/auth/login")
def login(payload: dict = Body(...)):
    print("USER_PASSWORD:", os.getenv("USER_PASSWORD"))
    print("ADMIN_PASSWORD:", os.getenv("ADMIN_PASSWORD"))
    password = payload.get("password")
    if password == os.getenv("ADMIN_PASSWORD"):
        return {"role": "admin", "token": os.getenv("ADMIN_TOKEN")}
    elif password == os.getenv("USER_PASSWORD"):
        return {"role": "user", "token": os.getenv("USER_TOKEN")}
    raise HTTPException(status_code=401, detail="Invalid password")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    auth_header = request.headers.get("Authorization")
    admin_token = os.getenv("ADMIN_TOKEN")
    user_token = os.getenv("USER_TOKEN")

    if path.startswith("/resources") or path.startswith("/static"):
        return await call_next(request)
    
    if path.startswith("/api") and not path.startswith("/api/auth/login"):
        # Admin-only routes
        if path.startswith("/api/indexing") or path.startswith("/api/document"):
            if auth_header != admin_token:
                return JSONResponse(status_code=403, content={"detail": "Admin token required"})
        # Chat routes accessible to user or admin
        elif path.startswith("/api/chat") or path.startswith("/api/chat_session"):
            if auth_header not in [user_token, admin_token]:
                return JSONResponse(status_code=403, content={"detail": "Valid user or admin token required"})
        else:
            if auth_header != admin_token:
                return JSONResponse(status_code=403, content={"detail": "Admin token required"})

    return await call_next(request)


# ===== Router setup =====
app.include_router(indexing.router, prefix="/api/indexing")
app.include_router(chat_session.router, prefix="/api/chat_session")
app.include_router(document.router, prefix="/api/document")
app.include_router(chat.router, prefix="/api/chat")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Catch-all for SPA routing =====
@app.get("/{full_path:path}")
def catch_all(full_path: str):
    target = Path("static/browser") / full_path
    if target.exists() and target.is_file():
        return FileResponse(target)
    return FileResponse(Path("static/browser/index.html"))
