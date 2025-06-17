
from dotenv import load_dotenv
load_dotenv()
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import requests
import os
import faiss
import pickle
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# === Setup ===

router = APIRouter()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")
RESOURCE_DIR = Path("./resources")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]  # type: ignore
sessions = db["chat_sessions"]
model = SentenceTransformer("all-MiniLM-L6-v2")

def call_ollama(messages: list, model=OLLAMA_MODEL):
    # url = "http://192.168.137.252:11434/api/chat"
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # log response if empty
        if "message" not in data:
            print(" Ollama response missing 'message':", data)
            return "[No reply returned from Ollama]"

        return data["message"].get("content", "").strip()

    except Exception as e:
        print("Ollama request failed:", e)
        return "[Ollama error: could not generate response]"


# === Routes ===
@router.get("/get_session/{session_id}")
def get_session(session_id: str):
    chat = sessions.find_one({"_id": session_id})
    if not chat:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return {
        "session_id": chat["_id"],
        "doc_ids": chat["doc_ids"],
        "messages": chat["messages"]
    }

@router.post("/resolve_docs")
def resolve_docs(data: dict = Body(...)):
    doc_ids = data.get("doc_ids", [])
    documents = db["documents"]
    found = list(documents.find(
        {"_id": {"$in": doc_ids}},
        {"_id": 1, "name": 1, "filename": 1}
    ))
    return found

@router.get("/list_sessions")
def list_sessions():
    all_sessions = sessions.find({}, {"_id": 1, "doc_ids": 1, "createdAt": 1})
    documents = db["documents"]

    enriched_sessions = []
    for s in all_sessions:
        doc_ids = s.get("doc_ids", [])
        doc_meta = list(documents.find(
            {"_id": {"$in": doc_ids}},
            {"_id": 1, "name": 1, "filename": 1}
        ))

        enriched_sessions.append({
            "id": s["_id"],
            "createdAt": s.get("createdAt"),
            "documents": doc_meta
        })

    return enriched_sessions

# === Embedding helper ===
def get_embedding(text: str):
    embedding = model.encode(text, convert_to_numpy=True)
    return np.array([embedding], dtype="float32")

def retrieve_context_from_faiss(doc_ids, query, top_k=5):
    combined_index = None
    all_text_chunks = []
    chunk_doc_map = []
    query_vec = get_embedding(query)

    for doc_id in doc_ids:
        doc_dir = RESOURCE_DIR / doc_id
        faiss_path = doc_dir / f"{doc_id}.faiss"
        pkl_path = doc_dir / f"{doc_id}.pkl"

        if not faiss_path.exists() or not pkl_path.exists():
            continue

        index = faiss.read_index(str(faiss_path))

        # ❗ Skip mismatched dimensions
        if index.d != query_vec.shape[1]:
            print(f"⚠️ Skipping {doc_id}: index.d={index.d} != query_vec.shape[1]={query_vec.shape[1]}")
            continue

        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
            text_chunks = meta.get("text", [])
            if isinstance(text_chunks, str):
                text_chunks = [text_chunks]

        for chunk in text_chunks:
            all_text_chunks.append(chunk)
            chunk_doc_map.append(doc_id)

        if combined_index is None:
            combined_index = index
        else:
            combined_index.merge_from(index)

    if combined_index is None or not all_text_chunks:
        return "No relevant document content found.", {}

    D, I = combined_index.search(query_vec, top_k)

    matched_chunks = []
    doc_score = {}

    for idx in I[0]:
        if 0 <= idx < len(all_text_chunks):
            matched_chunks.append(all_text_chunks[idx])
            doc_id = chunk_doc_map[idx]
            doc_score[doc_id] = doc_score.get(doc_id, 0) + 1

    return "\n\n".join(matched_chunks), doc_score

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
    new_session = sessions.find_one({"_id": session_id}, {"_id": 1, "doc_ids": 1, "createdAt": 1})
    documents = db["documents"]
    doc_meta = list(documents.find(
        {"_id": {"$in": new_session.get("doc_ids", [])}},
        {"_id": 1, "name": 1, "filename": 1}
    ))

    return {
        "session_id": session_id,
        "session": {
            "id": new_session["_id"],
            "createdAt": new_session["createdAt"],
            "documents": doc_meta
        }
    }

@router.post("/continue")
def continue_chat(data: dict = Body(...)):
    session_id = data.get("session_id")
    query = data.get("query")
    if not session_id or not query:
        return JSONResponse(status_code=400, content={"error": "session_id and query required"})

    chat = sessions.find_one({"_id": session_id})
    if not chat:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    messages = chat["messages"]
    messages.append({"role": "user", "content": query})

    fresh_doc_ids = chat.get("doc_ids", [])
    context_text, _ = retrieve_context_from_faiss(fresh_doc_ids, query)

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the following documents to answer accurately and precisely in shortest message possible:\n\n" + context_text},
        *messages
    ]

    reply = call_ollama(prompt_messages)
    messages.append({"role": "assistant", "content": reply})
    sessions.update_one({"_id": session_id}, {"$set": {"messages": messages}})

    return {"reply": reply, "messages": messages}
@router.post("/start2")
def start_chat_all_docs():
    session_id = str(uuid4())

    doc_ids = []
    doc_meta = []
    for path in RESOURCE_DIR.iterdir():
        doc_id = path.name
        faiss_file = path / f"{doc_id}.faiss"
        pkl_file = path / f"{doc_id}.pkl"
        if faiss_file.exists() and pkl_file.exists():
            doc_ids.append(doc_id)
            try:
                with open(pkl_file, "rb") as f:
                    meta = pickle.load(f)
                    doc_meta.append({
                        "_id": doc_id,
                        "name": meta.get("name", "N/A"),
                        "filename": meta.get("filename", "")
                    })
            except:
                doc_meta.append({"_id": doc_id, "name": "N/A", "filename": ""})

    sessions.insert_one({
        "_id": session_id,
        "doc_ids": doc_ids,
        "messages": [],
        "createdAt": datetime.utcnow()
    })

    return {
        "session_id": session_id,
        "session": {
            "id": session_id,
            "createdAt": datetime.utcnow(),
            "documents": doc_meta
        }
    }

@router.post("/continue2")
def continue_chat_all_docs(data: dict = Body(...)):
    session_id = data.get("session_id")
    query = data.get("query")
    if not session_id or not query:
        return JSONResponse(status_code=400, content={"error": "session_id and query required"})

    chat = sessions.find_one({"_id": session_id})
    if not chat:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    messages = chat["messages"]
    messages.append({"role": "user", "content": query})

    doc_ids = chat.get("doc_ids", [])
    context_text, doc_score = retrieve_context_from_faiss(doc_ids, query)

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the following documents to answer accurately and precisely. Prefer bullet points:\n\n" + context_text},
        *messages
    ]

    reply = call_ollama(prompt_messages)
    messages.append({"role": "assistant", "content": reply})
    sessions.update_one({"_id": session_id}, {"$set": {"messages": messages}})

    matched_docs = []
    for doc_id, score in sorted(doc_score.items(), key=lambda x: x[1], reverse=True):
        # meta_file = RESOURCE_DIR / doc_id / f"{doc_id}.pkl"
        meta_file = RESOURCE_DIR / f"{doc_id}.pkl"
        try:
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)
                matched_docs.append({
                    "doc_id": str(doc_id),
                    "doc_name": meta.get("name", "N/A"),
                    "link": f"/resources/{doc_id}/{meta.get('filename', '')}",
                    "score": score
                })
        except:
            matched_docs.append({
                "doc_id": str(doc_id),
                "doc_name": "N/A EXCEPTION",
                "link": f"/resources/{doc_id}/",
                "score": score
            })

    return {
        "reply": reply,
        "messages": messages,
        "matched_docs": matched_docs
    }
