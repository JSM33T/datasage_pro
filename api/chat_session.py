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
db = client[os.getenv("MONGO_DB")] # type: ignore
sessions = db["chat_sessions"]
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.get("/get_session/{session_id}")
def get_session(session_id: str):
    chat = sessions.find_one({ "_id": session_id })
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
        { "_id": { "$in": doc_ids } },
        { "_id": 1, "name": 1, "filename": 1 }
    ))
    return found

@router.get("/list_sessions")
def list_sessions():
    all_sessions = sessions.find({}, {"_id": 1, "doc_ids": 1, "createdAt": 1})

    # Access documents collection
    documents = db["documents"]

    enriched_sessions = []
    for s in all_sessions:
        doc_ids = s.get("doc_ids", [])
        doc_meta = list(documents.find(
            { "_id": { "$in": doc_ids } },
            { "_id": 1, "name": 1, "filename": 1 }
        ))

        enriched_sessions.append({
            "id": s["_id"],
            "createdAt": s.get("createdAt"),
            "documents": doc_meta
        })

    return enriched_sessions

    
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
    all_text_chunks = []

    for doc_id in doc_ids:
        doc_dir = RESOURCE_DIR / doc_id
        faiss_path = doc_dir / f"{doc_id}.faiss"
        pkl_path = doc_dir / f"{doc_id}.pkl"

        if not faiss_path.exists() or not pkl_path.exists():
            continue

        index = faiss.read_index(str(faiss_path))
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
            text_chunks = meta.get("text", [])
            if isinstance(text_chunks, str):
                text_chunks = [text_chunks]

        all_text_chunks.extend(text_chunks)

        if combined_index is None:
            combined_index = index
        else:
            combined_index.merge_from(index)

    if combined_index is None or not all_text_chunks:
        return "No relevant document content found."

    query_vec = get_embedding(query)
    D, I = combined_index.search(query_vec, top_k)

    matched_chunks = []
    for idx in I[0]:
        if 0 <= idx < len(all_text_chunks):
            matched_chunks.append(all_text_chunks[idx])

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
    new_session = sessions.find_one({ "_id": session_id }, {"_id": 1, "doc_ids": 1, "createdAt": 1 })
    documents = db["documents"]
    doc_meta = list(documents.find(
		{ "_id": { "$in": new_session.get("doc_ids", []) } }, # type: ignore
		{ "_id": 1, "name": 1, "filename": 1 }
	))

    return {
		"session_id": session_id,
		"session": {
			"id": new_session["_id"], # type: ignore
			"createdAt": new_session["createdAt"], # type: ignore
			"documents": doc_meta
		}
	}

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

    # Fetch context from FAISS + .pkl
    #context_text = retrieve_context_from_faiss(chat["doc_ids"], query)
    fresh_doc_ids = sessions.find_one({"_id": session_id}, {"doc_ids": 1}).get("doc_ids", []) # type: ignore
    context_text = retrieve_context_from_faiss(fresh_doc_ids, query)


    # Call GPT
    gpt_response = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the following documents to answer accurately and precisely wihtout hitting around the bush :\n\n" + context_text
            },
            *messages
        ]
    )

    reply = gpt_response.choices[0].message.content.strip() # type: ignore
    messages.append({ "role": "assistant", "content": reply })

    sessions.update_one({ "_id": session_id }, { "$set": { "messages": messages } })

    return {
        "reply": reply,
        "messages": messages
    }
