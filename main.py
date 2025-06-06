from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from pathlib import Path

from api import document,indexing,chat,chat_session

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse(Path("static/index.html"))


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
