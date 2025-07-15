from fastapi import APIRouter, Body, Request
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
load_dotenv(override=True)
router = APIRouter()

RESOURCE_DIR = Path("./resources")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")] # type: ignore
sessions = db["chat_sessions"]
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# === Get session details for UI compatibility ===
@router.get("/get_session/{session_id}")
def get_session(session_id: str, request: Request):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    
    # Find session by ID and ensure it belongs to the user
    chat = sessions.find_one({ "_id": session_id, "user_id": user_id })
    if not chat:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    
    # Get document metadata with download links
    documents = db["documents"]
    doc_ids = chat.get("doc_ids", [])
    doc_meta = list(documents.find(
        { "_id": { "$in": doc_ids } },
        { "_id": 1, "name": 1, "filename": 1 }
    ))
    
    # Add download links to documents
    for doc in doc_meta:
        doc["download_link"] = f"/api/document/download/{doc['_id']}/{doc['filename']}"
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    
    return {
        "session_id": chat["_id"],
        "doc_ids": chat["doc_ids"],
        "documents": doc_meta,
        "messages": chat["messages"]
    }


# === API: Clear all chat sessions ===
@router.post("/clear_all_sessions")
def clear_all_sessions(request: Request):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    
    # Only clear sessions belonging to the user
    result = sessions.delete_many({"user_id": user_id})
    return {"status": "success", "message": f"Cleared {result.deleted_count} chat sessions for user {user_id}."}

# === API: Clear messages for a specific session ===
@router.post("/clear_session/{session_id}")
def clear_session(session_id: str, request: Request):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    
    # Only clear if session belongs to the user
    result = sessions.update_one(
        {"_id": session_id, "user_id": user_id}, 
        {"$set": {"messages": []}}
    )
    if result.matched_count == 0:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return {"status": "success", "message": f"Session {session_id} messages cleared."}

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
    chunk_doc_map = []

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

        for chunk in text_chunks:
            all_text_chunks.append(chunk)
            chunk_doc_map.append(doc_id)

        if combined_index is None:
            combined_index = index
        else:
            combined_index.merge_from(index)

    if combined_index is None or not all_text_chunks:
        return "No relevant document content found.", {}

    query_vec = get_embedding(query)
    D, I = combined_index.search(query_vec, top_k)

    matched_chunks = []
    doc_score = {}

    for i, idx in enumerate(I[0]):
        if 0 <= idx < len(all_text_chunks):
            matched_chunks.append(all_text_chunks[idx])
            doc_id = chunk_doc_map[idx]
            distance = float(D[0][i])
            score = round(1 / (distance + 1e-6), 2)
            doc_score[doc_id] = doc_score.get(doc_id, 0.0) + score

    return "\n\n".join(matched_chunks), doc_score

# === API: List sessions (excluding master/all-docs session) ===
@router.get("/list_sessions")
def list_sessions(request: Request):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    
    # Get only sessions for the current user
    all_sessions = sessions.find({"user_id": user_id}, {"_id": 1, "doc_ids": 1, "createdAt": 1})
    documents = db["documents"]

    enriched_sessions = []
    all_doc_ids = [doc["_id"] for doc in documents.find({}, {"_id": 1})]
    for s in all_sessions:
        doc_ids = s.get("doc_ids", [])
        # Exclude master session (all docs)
        if set(doc_ids) == set(all_doc_ids):
            continue
        doc_meta = list(documents.find(
            { "_id": { "$in": doc_ids } },
            { "_id": 1, "name": 1, "filename": 1 }
        ))
        # Add download links to documents
        for doc in doc_meta:
            doc["download_link"] = f"/api/document/download/{doc['_id']}/{doc['filename']}"
            doc["id"] = str(doc["_id"])
            del doc["_id"]
        
        enriched_sessions.append({
            "id": s["_id"],
            "createdAt": s.get("createdAt"),
            "documents": doc_meta,
            "doc_ids": doc_ids
        })
    return enriched_sessions

# === API: Update chat session context ===
@router.post("/update_context")
def update_context(data: dict = Body(...), request: Request = None):
    user_id = request.state.user.get('username')
    session_id = data.get("session_id")
    doc_ids = data.get("doc_ids")

    if not session_id or not doc_ids:
        return JSONResponse(status_code=400, content={"error": "session_id and doc_ids are required"})

    result = sessions.update_one(
        {"_id": session_id, "user_id": user_id},
        {"$set": {"doc_ids": doc_ids}}
    )

    if result.matched_count == 0:
        return JSONResponse(status_code=404, content={"error": "Session not found or not owned by user"})

    return {"status": "success", "message": "Session context updated successfully."}


# === API: Start chat session with selected docs ===
@router.post("/start")
def start_chat(data: dict = Body(...), request: Request = None):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    doc_ids = data.get("doc_ids")
    if not doc_ids:
        return JSONResponse(status_code=400, content={"error": "doc_ids required"})

    session_id = str(uuid4())
    sessions.insert_one({
        "_id": session_id,
        "user_id": user_id,  # Associate session with user
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
    
    # Add download links to documents
    for doc in doc_meta:
        doc["download_link"] = f"/api/document/download/{doc['_id']}/{doc['filename']}"
        doc["id"] = str(doc["_id"])
        del doc["_id"]

    return {
        "session_id": session_id,
        "session": {
            "id": new_session["_id"], # type: ignore
            "createdAt": new_session["createdAt"], # type: ignore
            "documents": doc_meta
        }
    }

# === API: Continue chat in a session ===
@router.post("/continue")
def continue_chat(data: dict = Body(...), request: Request = None):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    session_id = data.get("session_id")
    query = data.get("query")
    if not session_id or not query:
        return JSONResponse(status_code=400, content={"error": "session_id and query required"})

    # Find session by ID and ensure it belongs to the user
    chat = sessions.find_one({ "_id": session_id, "user_id": user_id })
    if not chat:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    messages = chat["messages"]
    messages.append({ "role": "user", "content": query })

    # Fetch context from FAISS + .pkl - include recent conversation for better context retrieval
    doc_ids = chat.get("doc_ids", [])
    
    # Enhance query with recent conversation context for better retrieval
    recent_messages = messages[-6:]  # Last 3 exchanges (user + assistant)
    conversation_context = ""
    if len(recent_messages) > 2:
        conversation_context = " ".join([msg["content"] for msg in recent_messages[-4:]])
        enhanced_query = f"{query} {conversation_context}"
    else:
        enhanced_query = query
    
    context_text, doc_score = retrieve_context_from_faiss(doc_ids, enhanced_query)

    # Strict anti-hallucination: if no context or all scores below threshold, reply "I don't know." except for greetings
    SIMILARITY_THRESHOLD = 0.75
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    user_query = query.strip().lower()
    if (not context_text or all(score < SIMILARITY_THRESHOLD for score in doc_score.values())):
        if any(greet in user_query for greet in greetings):
            reply = "Hello! How can I assist you with your selected documents?"
        else:
            reply = "I don't know."
    else:
        # Create system prompt with current context
        system_prompt = (
            "You are a helpful document context assistant. Answer using the information in the provided context below and maintain conversation continuity. "
            "Reference previous parts of our conversation when relevant. "
            "If the answer is not present in the context, first combine it with the comtext of the document present , and in the last resort, reply with 'No relevant document found for your query.'\n\nContext:\n" + context_text
        )
        
        # Limit message history but keep enough for context (increase to 20 messages)
        MAX_MESSAGES = 20
        conversation_messages = messages[-MAX_MESSAGES:]
        
        gpt_response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                *conversation_messages
            ]
        )
        reply = gpt_response.choices[0].message.content.strip()  # type: ignore

    messages.append({ "role": "assistant", "content": reply })
    sessions.update_one({ "_id": session_id, "user_id": user_id }, { "$set": { "messages": messages } })

    # Sort docs by relevance score
    sorted_docs = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, _ in sorted_docs]

    documents = db["documents"]
    docs_cursor = documents.find(
        { "_id": { "$in": top_doc_ids } },
        { "_id": 1, "name": 1, "filename": 1 }
    )
    doc_map = { str(d["_id"]): d for d in docs_cursor }

    matched_docs = [
        {
            "doc_id": str(doc_id),
            "doc_name": doc_map.get(str(doc_id), {}).get("name", ""),
            "link": f"/resources/{doc_id}/{doc_map.get(str(doc_id), {}).get('filename', '')}",
            "download_link": f"/api/document/download/{doc_id}/{doc_map.get(str(doc_id), {}).get('filename', '')}",
            "score": doc_score[doc_id]
        }
        for doc_id in top_doc_ids if str(doc_id) in doc_map
    ]

    return {
        "reply": reply,
        "messages": messages,
        "matched_docs": matched_docs
    }
