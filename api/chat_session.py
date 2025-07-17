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
load_dotenv(override=True)  # force reload
router = APIRouter()

RESOURCE_DIR = Path("./resources")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")] # type: ignore
sessions = db["chat_sessions"]
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.get("/get_session/{session_id}")
def get_session(session_id: str, request: Request):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    print("========================================");
    print(openai_client.api_key);
    print("========================================");
    
    # Find session by ID and ensure it belongs to the user
    chat = sessions.find_one({ "_id": session_id, "user_id": user_id })
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
def list_sessions(request: Request):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    
    # Get only sessions for the current user
    all_sessions = sessions.find({"user_id": user_id}, {"_id": 1, "doc_ids": 1, "createdAt": 1})

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
    chunk_doc_map = []
    doc_names = {}

    # Get document names for better context
    documents = db["documents"]
    doc_cursor = documents.find({"_id": {"$in": doc_ids}}, {"_id": 1, "name": 1})
    for doc in doc_cursor:
        doc_names[doc["_id"]] = doc["name"]

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
            doc_id = chunk_doc_map[idx]
            doc_name = doc_names.get(doc_id, f"Document_{doc_id}")
            chunk_with_source = f"[From {doc_name}]:\n{all_text_chunks[idx]}"
            matched_chunks.append(chunk_with_source)
            distance = float(D[0][i])
            score = round(1 / (distance + 1e-6), 2)
            doc_score[doc_id] = doc_score.get(doc_id, 0.0) + score

    return "\n\n".join(matched_chunks), doc_score

# === API: Start chat session ===
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

    # Fetch context from FAISS + .pkl
    fresh_doc_ids = sessions.find_one({"_id": session_id, "user_id": user_id}, {"doc_ids": 1}).get("doc_ids", []) # type: ignore
    context_text, doc_score = retrieve_context_from_faiss(fresh_doc_ids, query)

    # Log context for debugging
    print("\n[DEBUG] Context chunks sent to LLM:\n", context_text, "\n")
    print(f"[DEBUG] Doc scores: {doc_score}")

    # HUMANIZED FALLBACK: If score > 5%, extract content and use AI to provide coherent response
    if doc_score and max(doc_score.values()) > 0.05:  # 5% threshold
        max_score = max(doc_score.values())
        top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
        
        # Get document name
        documents = db["documents"]
        doc_info = documents.find_one({"_id": top_doc_id}, {"name": 1})
        doc_name = doc_info["name"] if doc_info else f"Document_{top_doc_id}"
        
        print(f"[DEBUG] Score {max_score:.3f} > 5%, preparing fallback for {doc_name}")
        
        # Check if we should do direct extraction (poor context or likely refusal)
        should_extract = (not context_text or 
                         "No relevant document content found" in context_text or 
                         len(context_text.strip()) < 50)
        
        if should_extract:
            print(f"[DEBUG] Performing AI-processed extraction from {doc_name}")
            
            # Get raw chunks from the highest scoring document
            doc_dir = RESOURCE_DIR / top_doc_id
            pkl_path = doc_dir / f"{top_doc_id}.pkl"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        meta = pickle.load(f)
                        text_chunks = meta.get("text", [])
                        if isinstance(text_chunks, str):
                            text_chunks = [text_chunks]
                    
                    if text_chunks:
                        # Extract first few meaningful chunks
                        meaningful_chunks = []
                        for chunk in text_chunks[:10]:  # First 10 chunks
                            if chunk.strip() and len(chunk.strip()) > 20:  # Skip very short chunks
                                meaningful_chunks.append(chunk.strip())
                            if len(meaningful_chunks) >= 3:  # Get 3 good chunks
                                break
                        
                        if meaningful_chunks:
                            # Use OpenAI to process the chunks and provide a coherent answer
                            combined_content = "\n\n".join(meaningful_chunks)
                            
                            processing_prompt = f"""
You are a helpful assistant. The user asked: "{query}"

I found relevant content in the {doc_name} document. Please read through this content and provide a helpful, coherent answer to the user's question based on what you find:

DOCUMENT CONTENT:
{combined_content}

Please provide a clear, direct answer based on the content above. If the content doesn't fully answer the question, explain what information is available and how it relates to the query.
"""
                            
                            gpt_response = openai_client.chat.completions.create(
                                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant. Read the provided document content and answer the user's question based on what you find. Be direct and helpful."
                                    },
                                    {
                                        "role": "user",
                                        "content": processing_prompt
                                    }
                                ]
                            )
                            
                            processed_reply = gpt_response.choices[0].message.content.strip()
                            reply = f"{processed_reply}\n\n*Source: {doc_name} (relevance score: {max_score:.1f}%)*"
                            
                            print(f"[DEBUG] AI-processed extraction successful, {len(meaningful_chunks)} chunks processed")
                            
                            # Update messages and return early
                            messages.append({ "role": "assistant", "content": reply })
                            sessions.update_one({ "_id": session_id, "user_id": user_id }, { "$set": { "messages": messages } })
                            return {
                                "reply": reply,
                                "messages": messages
                            }
                except Exception as e:
                    print(f"[DEBUG] Error during AI-processed extraction: {e}")

    # Prompt engineering: restrict LLM to context only
    # system_prompt = (
    #     "You are a helpful assistant. Answer ONLY using the information in the provided context below. "
    #     "If the answer is not present in the context, reply with 'No revelant document found.'\n\nContext:\n" + context_text
    # )
    system_prompt = (
            "You are a helpful document context assistant. Answer using the information in the provided context below and maintain conversation continuity. "
            "Reference previous parts of our conversation when relevant. "
            "If the answer is not present in the context, first combine it with the comtext of the document present and try to combine the query with the first document received , and in the last resort, reply with 'No relevant document found for your query.'\n\nContext:\n" + context_text
        )

    gpt_response = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ]
    )

    reply = gpt_response.choices[0].message.content.strip() # type: ignore
    
    # Post-processing check: If LLM still refuses despite high score, force AI-processed response
    if (doc_score and max(doc_score.values()) > 0.05 and 
        ("no relevant" in reply.lower() or "not found" in reply.lower() or "don't have" in reply.lower())):
        
        max_score = max(doc_score.values())
        top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
        
        print(f"[DEBUG] LLM refused despite score {max_score:.3f}, forcing AI-processed extraction")
        
        # Direct extraction from highest scoring document
        doc_dir = RESOURCE_DIR / top_doc_id
        pkl_path = doc_dir / f"{top_doc_id}.pkl"
        
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    meta = pickle.load(f)
                    text_chunks = meta.get("text", [])
                    if isinstance(text_chunks, str):
                        text_chunks = [text_chunks]
                
                if text_chunks:
                    # Get multiple meaningful chunks for better processing
                    meaningful_chunks = []
                    for chunk in text_chunks[:10]:
                        if chunk.strip() and len(chunk.strip()) > 30:
                            meaningful_chunks.append(chunk.strip())
                        if len(meaningful_chunks) >= 3:
                            break
                    
                    if meaningful_chunks:
                        # Get document name
                        documents = db["documents"]
                        doc_info = documents.find_one({"_id": top_doc_id}, {"name": 1})
                        doc_name = doc_info["name"] if doc_info else f"Document_{top_doc_id}"
                        
                        # Use OpenAI to process the chunks and provide a coherent answer
                        combined_content = "\n\n".join(meaningful_chunks)
                        
                        processing_prompt = f"""
You are a helpful assistant. The user asked: "{query}"

I found relevant content in the {doc_name} document. Please read through this content and provide a helpful, coherent answer to the user's question based on what you find:

DOCUMENT CONTENT:
{combined_content}

Please provide a clear, direct answer based on the content above. If the content doesn't fully answer the question, explain what information is available and how it relates to the query.
"""
                        
                        gpt_response = openai_client.chat.completions.create(
                            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant. Read the provided document content and answer the user's question based on what you find. Be direct and helpful."
                                },
                                {
                                    "role": "user",
                                    "content": processing_prompt
                                }
                            ]
                        )
                        
                        processed_reply = gpt_response.choices[0].message.content.strip()
                        reply = f"{processed_reply}\n\n*Source: {doc_name} (relevance score: {max_score:.1f}%)*"
                        
                        print(f"[DEBUG] Forced AI-processed extraction successful from {doc_name}")
            except Exception as e:
                print(f"[DEBUG] Error in forced AI-processed extraction: {e}")
    
    messages.append({ "role": "assistant", "content": reply })

    sessions.update_one({ "_id": session_id, "user_id": user_id }, { "$set": { "messages": messages } })

    return {
        "reply": reply,
        "messages": messages
    }


@router.post("/start2")
def start_chat_all_docs(request: Request):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    session_id = str(uuid4())
    documents = db["documents"]
    all_docs = list(documents.find({}, { "_id": 1, "name": 1, "filename": 1 }))
    doc_ids = [doc["_id"] for doc in all_docs]

    sessions.insert_one({
        "_id": session_id,
        "user_id": user_id,  # Associate session with user
        "doc_ids": doc_ids,
        "messages": [],
        "createdAt": datetime.utcnow()
    })

    return {
        "session_id": session_id,
        "session": {
            "id": session_id,
            "createdAt": datetime.utcnow(),
            "documents": all_docs
        }
    }

@router.post("/continue2")
def continue_chat_all_docs(data: dict = Body(...), request: Request = None):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    from tiktoken import encoding_for_model

    def count_tokens(text, model="gpt-3.5-turbo"):
        enc = encoding_for_model(model)
        return len(enc.encode(text))

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

    all_doc_ids = [doc["_id"] for doc in db["documents"].find({}, { "_id": 1 })]
    
    # === Retrieve context ===
    context_text, doc_score = retrieve_context_from_faiss(all_doc_ids, query)

    # Log context for debugging
    print("\n[DEBUG] Context chunks sent to LLM:\n", context_text, "\n")

    # === Token count trimming for context_text at chunk boundaries ===
    MAX_CONTEXT_TOKENS = 3000
    context_chunks = context_text.split("\n\n")
    trimmed_context = []
    total_tokens = 0
    for chunk in context_chunks:
        chunk_tokens = count_tokens(chunk)
        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        trimmed_context.append(chunk)
        total_tokens += chunk_tokens
    context_text_final = "\n\n".join(trimmed_context)

    # === Limit message history ===
    MAX_MESSAGES = 10
    messages = messages[-MAX_MESSAGES:]

    # Strict anti-hallucination: if no context or all scores below threshold, try AI processing fallback
    SIMILARITY_THRESHOLD = 0.75
    if not context_text_final or all(score < SIMILARITY_THRESHOLD for score in doc_score.values()):
        # Check if we have any document with score > 5% for AI processing
        if doc_score and max(doc_score.values()) > 0.05:
            max_score = max(doc_score.values())
            top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
            
            print(f"[DEBUG] Low context but score {max_score:.3f} > 5%, trying AI processing")
            
            # Direct extraction from highest scoring document
            doc_dir = RESOURCE_DIR / top_doc_id
            pkl_path = doc_dir / f"{top_doc_id}.pkl"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        meta = pickle.load(f)
                        text_chunks = meta.get("text", [])
                        if isinstance(text_chunks, str):
                            text_chunks = [text_chunks]
                    
                    if text_chunks:
                        # Get first meaningful chunks
                        meaningful_chunks = []
                        for chunk in text_chunks[:10]:
                            if chunk.strip() and len(chunk.strip()) > 30:
                                meaningful_chunks.append(chunk.strip())
                            if len(meaningful_chunks) >= 3:
                                break
                        
                        if meaningful_chunks:
                            # Get document name
                            documents = db["documents"]
                            doc_info = documents.find_one({"_id": top_doc_id}, {"name": 1})
                            doc_name = doc_info["name"] if doc_info else f"Document_{top_doc_id}"
                            
                            # Use OpenAI to process the chunks and provide a coherent answer
                            combined_content = "\n\n".join(meaningful_chunks)
                            
                            processing_prompt = f"""
You are a helpful assistant. The user asked: "{query}"

I found relevant content in the {doc_name} document. Please read through this content and provide a helpful, coherent answer to the user's question based on what you find:

DOCUMENT CONTENT:
{combined_content}

Please provide a clear, direct answer based on the content above. If the content doesn't fully answer the question, explain what information is available and how it relates to the query.
"""
                            
                            gpt_response = openai_client.chat.completions.create(
                                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant. Read the provided document content and answer the user's question based on what you find. Be direct and helpful."
                                    },
                                    {
                                        "role": "user",
                                        "content": processing_prompt
                                    }
                                ]
                            )
                            
                            processed_reply = gpt_response.choices[0].message.content.strip()
                            reply = f"{processed_reply}\n\n*Source: {doc_name} (relevance score: {max_score:.1f}%)*"
                            
                            print(f"[DEBUG] AI processing successful from {doc_name}")
                        else:
                            reply = "I don't know."
                    else:
                        reply = "I don't know."
                except Exception as e:
                    print(f"[DEBUG] Error in AI processing: {e}")
                    reply = "I don't know."
            else:
                reply = "I don't know."
        else:
            reply = "I don't know."
    else:
        system_prompt = (
            "You are a helpful document context assistant. Answer ONLY using the information in the provided context below. "
            "If the answer is not present in the context, reply with 'No relevant document found for your query.'\n\nContext:\n" + context_text_final
        )
        gpt_response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                *messages
            ]
        )
        reply = gpt_response.choices[0].message.content.strip()  # type: ignore
        
        # Post-processing check: If LLM still refuses despite high score, force AI processing
        if (doc_score and max(doc_score.values()) > 0.05 and 
            ("no relevant" in reply.lower() or "not found" in reply.lower() or "don't have" in reply.lower())):
            
            max_score = max(doc_score.values())
            top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
            
            print(f"[DEBUG] LLM refused despite score {max_score:.3f}, forcing AI processing")
            
            # Direct extraction from highest scoring document
            doc_dir = RESOURCE_DIR / top_doc_id
            pkl_path = doc_dir / f"{top_doc_id}.pkl"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        meta = pickle.load(f)
                        text_chunks = meta.get("text", [])
                        if isinstance(text_chunks, str):
                            text_chunks = [text_chunks]
                    
                    if text_chunks:
                        # Get first meaningful chunks
                        meaningful_chunks = []
                        for chunk in text_chunks[:10]:
                            if chunk.strip() and len(chunk.strip()) > 30:
                                meaningful_chunks.append(chunk.strip())
                            if len(meaningful_chunks) >= 3:
                                break
                        
                        if meaningful_chunks:
                            # Get document name
                            documents = db["documents"]
                            doc_info = documents.find_one({"_id": top_doc_id}, {"name": 1})
                            doc_name = doc_info["name"] if doc_info else f"Document_{top_doc_id}"
                            
                            # Use OpenAI to process the chunks and provide a coherent answer
                            combined_content = "\n\n".join(meaningful_chunks)
                            
                            processing_prompt = f"""
You are a helpful assistant. The user asked: "{query}"

I found relevant content in the {doc_name} document. Please read through this content and provide a helpful, coherent answer to the user's question based on what you find:

DOCUMENT CONTENT:
{combined_content}

Please provide a clear, direct answer based on the content above. If the content doesn't fully answer the question, explain what information is available and how it relates to the query.
"""
                            
                            gpt_response = openai_client.chat.completions.create(
                                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant. Read the provided document content and answer the user's question based on what you find. Be direct and helpful."
                                    },
                                    {
                                        "role": "user",
                                        "content": processing_prompt
                                    }
                                ]
                            )
                            
                            processed_reply = gpt_response.choices[0].message.content.strip()
                            reply = f"{processed_reply}\n\n*Source: {doc_name} (relevance score: {max_score:.1f}%)*"
                            
                            print(f"[DEBUG] Forced AI processing successful from {doc_name}")
                except Exception as e:
                    print(f"[DEBUG] Error in forced AI processing: {e}")

    messages.append({ "role": "assistant", "content": reply })
    sessions.update_one({ "_id": session_id, "user_id": user_id }, { "$set": { "messages": messages } })

    # === Sort docs by relevance score ===
    sorted_docs = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, _ in sorted_docs]

    docs_cursor = db["documents"].find(
        { "_id": { "$in": top_doc_ids } },
        { "_id": 1, "name": 1, "filename": 1 }
    )
    doc_map = { str(d["_id"]): d for d in docs_cursor }

    matched_docs = [
        {
            "doc_id": str(doc_id),
            "doc_name": doc_map.get(str(doc_id), {}).get("name", ""),
            "link": f"/resources/{doc_id}/{doc_map.get(str(doc_id), {}).get('filename', '')}",
            "score": round(doc_score[doc_id], 2)
        }
        for doc_id in top_doc_ids if str(doc_id) in doc_map
    ]

    return {
        "reply": reply,
        "messages": messages,
        "matched_docs": matched_docs
    }


@router.post("/continue3")
def continue_chat_all_docs_v3(data: dict = Body(...), request: Request = None):
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

    all_doc_ids = [doc["_id"] for doc in db["documents"].find({}, { "_id": 1 })]
    
    # Updated: retrieve context and doc_score mapping
    context_text, doc_score = retrieve_context_from_faiss(all_doc_ids, query)

    # HUMANIZED FALLBACK: If score > 5% but poor context, try AI processing
    if (doc_score and max(doc_score.values()) > 0.05 and 
        (not context_text or "No relevant document content found" in context_text or len(context_text.strip()) < 50)):
        
        max_score = max(doc_score.values())
        top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
        
        print(f"[DEBUG] Score {max_score:.3f} > 5%, trying AI processing for continue3")
        
        # Direct extraction from highest scoring document
        doc_dir = RESOURCE_DIR / top_doc_id
        pkl_path = doc_dir / f"{top_doc_id}.pkl"
        
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    meta = pickle.load(f)
                    text_chunks = meta.get("text", [])
                    if isinstance(text_chunks, str):
                        text_chunks = [text_chunks]
                
                if text_chunks:
                    # Extract first few meaningful chunks
                    meaningful_chunks = []
                    for chunk in text_chunks[:10]:  # First 10 chunks
                        if chunk.strip() and len(chunk.strip()) > 20:  # Skip very short chunks
                            meaningful_chunks.append(chunk.strip())
                        if len(meaningful_chunks) >= 3:  # Get 3 good chunks
                            break
                    
                    if meaningful_chunks:
                        # Get document name
                        documents = db["documents"]
                        doc_info = documents.find_one({"_id": top_doc_id}, {"name": 1})
                        doc_name = doc_info["name"] if doc_info else f"Document_{top_doc_id}"
                        
                        # Use OpenAI to process the chunks and provide a coherent answer
                        combined_content = "\n\n".join(meaningful_chunks)
                        
                        processing_prompt = f"""
You are a helpful assistant. The user asked: "{query}"

I found relevant content in the {doc_name} document. Please read through this content and provide a helpful, coherent answer to the user's question based on what you find:

DOCUMENT CONTENT:
{combined_content}

Please provide a clear, direct answer based on the content above. If the content doesn't fully answer the question, explain what information is available and how it relates to the query.
"""
                        
                        gpt_response = openai_client.chat.completions.create(
                            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant. Read the provided document content and answer the user's question based on what you find. Be direct and helpful."
                                },
                                {
                                    "role": "user",
                                    "content": processing_prompt
                                }
                            ]
                        )
                        
                        processed_reply = gpt_response.choices[0].message.content.strip()
                        reply = f"{processed_reply}\n\n*Source: {doc_name} (relevance score: {max_score:.1f}%)*"
                        
                        print(f"[DEBUG] AI processing successful for continue3, {len(meaningful_chunks)} chunks processed")
                        
                        # Update messages and return early
                        messages.append({ "role": "assistant", "content": reply })
                        sessions.update_one({ "_id": session_id, "user_id": user_id }, { "$set": { "messages": messages } })
                        
                        # Sort docs by relevance score
                        sorted_docs = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
                        top_doc_ids = [doc_id for doc_id, _ in sorted_docs]

                        docs_cursor = db["documents"].find(
                            { "_id": { "$in": top_doc_ids } },
                            { "_id": 1, "name": 1, "filename": 1 }
                        )
                        doc_map = { str(d["_id"]): d for d in docs_cursor }

                        matched_docs = [
                            {
                                "doc_id": str(doc_id),
                                "doc_name": doc_map.get(str(doc_id), {}).get("name", ""),
                                "link": f"/resources/{doc_id}/{doc_map.get(str(doc_id), {}).get('filename', '')}",
                                "score": round(doc_score[doc_id], 2)
                            }
                            for doc_id in top_doc_ids if str(doc_id) in doc_map
                        ]

                        return {
                            "reply": reply,
                            "messages": messages,
                            "matched_docs": matched_docs
                        }
            except Exception as e:
                print(f"[DEBUG] Error during AI processing in continue3: {e}")

    gpt_response = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the following documents to answer accurately and precisely wihtout diverting from the context:\n\n" + context_text
            },
            *messages
        ]
    )

    reply = gpt_response.choices[0].message.content.strip()  # type: ignore
    
    # Post-processing check: If LLM still refuses despite high score, force AI processing
    if (doc_score and max(doc_score.values()) > 0.05 and 
        ("no relevant" in reply.lower() or "not found" in reply.lower() or "don't have" in reply.lower())):
        
        max_score = max(doc_score.values())
        top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
        
        print(f"[DEBUG] LLM refused despite score {max_score:.3f}, forcing AI processing in continue3")
        
        # Direct extraction from highest scoring document
        doc_dir = RESOURCE_DIR / top_doc_id
        pkl_path = doc_dir / f"{top_doc_id}.pkl"
        
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    meta = pickle.load(f)
                    text_chunks = meta.get("text", [])
                    if isinstance(text_chunks, str):
                        text_chunks = [text_chunks]
                
                if text_chunks:
                    # Get first meaningful chunks
                    meaningful_chunks = []
                    for chunk in text_chunks[:10]:
                        if chunk.strip() and len(chunk.strip()) > 30:
                            meaningful_chunks.append(chunk.strip())
                        if len(meaningful_chunks) >= 3:
                            break
                    
                    if meaningful_chunks:
                        # Get document name
                        documents = db["documents"]
                        doc_info = documents.find_one({"_id": top_doc_id}, {"name": 1})
                        doc_name = doc_info["name"] if doc_info else f"Document_{top_doc_id}"
                        
                        # Use OpenAI to process the chunks and provide a coherent answer
                        combined_content = "\n\n".join(meaningful_chunks)
                        
                        processing_prompt = f"""
You are a helpful assistant. The user asked: "{query}"

I found relevant content in the {doc_name} document. Please read through this content and provide a helpful, coherent answer to the user's question based on what you find:

DOCUMENT CONTENT:
{combined_content}

Please provide a clear, direct answer based on the content above. If the content doesn't fully answer the question, explain what information is available and how it relates to the query.
"""
                        
                        gpt_response = openai_client.chat.completions.create(
                            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant. Read the provided document content and answer the user's question based on what you find. Be direct and helpful."
                                },
                                {
                                    "role": "user",
                                    "content": processing_prompt
                                }
                            ]
                        )
                        
                        processed_reply = gpt_response.choices[0].message.content.strip()
                        reply = f"{processed_reply}\n\n*Source: {doc_name} (relevance score: {max_score:.1f}%)*"
                        
                        print(f"[DEBUG] Forced AI processing successful from {doc_name} in continue3")
            except Exception as e:
                print(f"[DEBUG] Error in forced AI processing in continue3: {e}")
    
    messages.append({ "role": "assistant", "content": reply })

    sessions.update_one({ "_id": session_id, "user_id": user_id }, { "$set": { "messages": messages } })

    # Sort docs by relevance score
    sorted_docs = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, _ in sorted_docs]

    docs_cursor = db["documents"].find(
        { "_id": { "$in": top_doc_ids } },
        { "_id": 1, "name": 1, "filename": 1 }
    )
    doc_map = { str(d["_id"]): d for d in docs_cursor }

    matched_docs = [
        {
            "doc_id": str(doc_id),
            "doc_name": doc_map.get(str(doc_id), {}).get("name", ""),
            "link": f"/resources/{doc_id}/{doc_map.get(str(doc_id), {}).get('filename', '')}",
            "score": round(doc_score[doc_id], 2)
        }
        for doc_id in top_doc_ids if str(doc_id) in doc_map
    ]

    return {
        "reply": reply,
        "messages": messages,
        "matched_docs": matched_docs
    }

def retrieve_context_from_faiss(doc_ids, query, top_k=5):
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
            distance = float(D[0][i])  # ensure native float
            #score = float(1 / (distance + 1e-6))  # convert to float
            score = round(1 / (float(distance) + 1e-6), 2)
            doc_score[doc_id] = doc_score.get(doc_id, 0.0) + score

    return "\n\n".join(matched_chunks), doc_score
