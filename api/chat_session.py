from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pymongo import MongoClient
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import openai
import os
import faiss
import pickle
import numpy as np

# === Setup ===
load_dotenv()
router = APIRouter()

RESOURCE_DIR = Path("./resources")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]
sessions = db["chat_sessions"]
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Embedding helper ===
def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array([response.data[0].embedding], dtype="float32")

# === Context retrieval from FAISS + .pkl ===
def retrieve_context_from_faiss(doc_ids, query, top_k=3):
    combined_index = None
    all_metadata = []

    for doc_id in doc_ids:
        doc_dir = RESOURCE_DIR / doc_id
        faiss_path = doc_dir / f"{doc_id}.faiss"
        pkl_path = doc_dir / f"{doc_id}.pkl"

        if not faiss_path.exists() or not pkl_path.exists():
            continue

        index = faiss.read_index(str(faiss_path))
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
            all_metadata.append(meta)

        if combined_index is None:
            combined_index = index
        else:
            combined_index.merge_from(index)

    if combined_index is None or not all_metadata:
        return "No relevant document content found."

    query_vec = get_embedding(query)
    D, I = combined_index.search(query_vec, top_k)

    matched_chunks = []
    for i in I[0]:
        for meta in all_metadata:
            text = meta.get("text", "")
            if text:
                matched_chunks.append(text)
                break

    return "\n\n".join(matched_chunks)

# === API: Start chat session ===
@router.post("/start")
def start_chat(data: dict = Body(...)):
    doc_ids = data.get("doc_ids")
    if not doc_ids:
        return JSONResponse(status_code=400, content={"error": "doc_ids required"})

    session_id = str(uuid4())
    sessions.insert_one({
        "_id": session_id,
        "doc_ids": doc_ids,
        "messages": [],
        "createdAt": datetime.utcnow()
    })
    return { "session_id": session_id }

# === API: Continue chat ===
@router.post("/continue")
def continue_chat(data: dict = Body(...)):
    session_id = data.get("session_id")
    query = data.get("query")
    if not session_id or not query:
        return JSONResponse(status_code=400, content={"error": "session_id and query required"})

    chat = sessions.find_one({ "_id": session_id })
    if not chat:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    messages = chat["messages"]
    messages.append({ "role": "user", "content": query })

    # 🧠 Fetch context from FAISS + .pkl
    context_text = retrieve_context_from_faiss(chat["doc_ids"], query)

    # 🧠 Call GPT
    gpt_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the following documents to answer accurately:\n\n" + context_text
            },
            *messages
        ]
    )

    reply = gpt_response.choices[0].message.content.strip()
    messages.append({ "role": "assistant", "content": reply })

    sessions.update_one({ "_id": session_id }, { "$set": { "messages": messages } })

    return {
        "reply": reply,
        "messages": messages
    }
