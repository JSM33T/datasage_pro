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
import re
from collections import Counter

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

# === Extract keywords from conversation ===
def extract_keywords_from_conversation(messages, current_query):
    """Extract important keywords from conversation history and current query"""
    # Combine all messages
    all_text = current_query + " "
    for msg in messages[-10:]:  # Last 10 messages
        all_text += msg.get("content", "") + " "
    
    # Clean and normalize text
    text = re.sub(r'[^\w\s]', ' ', all_text.lower())
    words = text.split()
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'where', 'when', 'why', 'how', 'who', 'which', 'all', 'any', 'some', 'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'here', 'there', 'then', 'up', 'out', 'if', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'from', 'please', 'help', 'can', 'tell', 'know', 'find', 'show', 'explain', 'describe', 'document', 'documents', 'information', 'content', 'text', 'file', 'files'
    }
    
    # Filter meaningful words (length > 2, not stop words)
    meaningful_words = [word for word in words if len(word) > 2 and word not in stop_words]
    
    # Count word frequency and get top keywords
    word_counts = Counter(meaningful_words)
    top_keywords = [word for word, count in word_counts.most_common(15)]
    
    return top_keywords

# === Enhanced context retrieval with keyword fallback ===
def retrieve_context_with_fallback(doc_ids, query, messages, top_k=8):
    """Retrieve context with multiple fallback strategies"""
    
    # Strategy 1: Original query
    context_text, doc_score = retrieve_context_from_faiss(doc_ids, query, top_k)
    
    # Check if we got good results
    if doc_score and max(doc_score.values()) > 0.4:
        return context_text, doc_score
    
    # Strategy 2: Enhanced query with conversation context
    recent_messages = messages[-8:]
    if len(recent_messages) > 2:
        conversation_context = " ".join([msg["content"] for msg in recent_messages[-6:]])
        enhanced_query = f"{conversation_context} {query}"
        context_text, doc_score = retrieve_context_from_faiss(doc_ids, enhanced_query, top_k)
        
        if doc_score and max(doc_score.values()) > 0.4:
            return context_text, doc_score
    
    # Strategy 3: Keyword-based fallback
    keywords = extract_keywords_from_conversation(messages, query)
    if keywords:
        # Try top 5 keywords first
        keyword_query = " ".join(keywords[:5])
        context_text_kw, doc_score_kw = retrieve_context_from_faiss(doc_ids, keyword_query, top_k)
        
        if doc_score_kw and max(doc_score_kw.values()) > 0.3:
            # Combine with original if we have both
            if context_text and "No relevant document content found" not in context_text:
                combined_context = context_text + "\n\n=== Additional context from keywords ===\n" + context_text_kw
                # Merge scores
                for doc_id, score in doc_score_kw.items():
                    doc_score[doc_id] = doc_score.get(doc_id, 0) + score * 0.8
                return combined_context, doc_score
            else:
                return context_text_kw, doc_score_kw
        
        # Strategy 4: Try with broader keywords if still no good results
        if len(keywords) > 5:
            broader_keyword_query = " ".join(keywords[:10])
            context_text_broad, doc_score_broad = retrieve_context_from_faiss(doc_ids, broader_keyword_query, top_k)
            
            if doc_score_broad and max(doc_score_broad.values()) > 0.25:
                return context_text_broad, doc_score_broad
    
    # Strategy 5: Individual document search as last resort
    all_contexts = []
    combined_scores = {}
    
    for doc_id in doc_ids[:5]:  # Limit to first 5 docs to avoid timeout
        doc_context, doc_score_individual = retrieve_context_from_faiss([doc_id], query, top_k=3)
        if doc_context and "No relevant document content found" not in doc_context:
            all_contexts.append(doc_context)
            for d_id, score in doc_score_individual.items():
                combined_scores[d_id] = combined_scores.get(d_id, 0) + score
    
    if all_contexts:
        return "\n\n=== Individual document search results ===\n".join(all_contexts), combined_scores
    
    # Return original results if all fallbacks fail
    return context_text, doc_score

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
    
    # Use enhanced context retrieval with fallback strategies
    context_text, doc_score = retrieve_context_with_fallback(doc_ids, query, messages, top_k=8)
    
    # If we have doc_score results, try to get additional context from top-ranked documents
    if doc_score:
        # Get top 3 documents by score
        top_doc_ids = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # For each top document, get additional context with the original query
        additional_contexts = []
        for doc_id, score in top_doc_ids:
            if score > 0.2:  # Lowered threshold for additional context
                additional_context, _ = retrieve_context_from_faiss([doc_id], query, top_k=3)
                if additional_context and "No relevant document content found" not in additional_context:
                    additional_contexts.append(f"\n=== Additional context from top-ranked document ===\n{additional_context}")
        
        # Combine original context with additional context from top documents
        if additional_contexts:
            context_text = context_text + "\n\n".join(additional_contexts)

    # Strict anti-hallucination: if no context or all scores below threshold, reply "I don't know." except for greetings
    SIMILARITY_THRESHOLD = 0.2  # Lowered threshold after multiple fallback attempts
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    user_query = query.strip().lower()
    
    # More lenient check after fallback attempts
    if (not context_text or 
        "No relevant document content found" in context_text or 
        (doc_score and all(score < SIMILARITY_THRESHOLD for score in doc_score.values()))):
        
        if any(greet in user_query for greet in greetings):
            reply = "Hello! How can I assist you with your selected documents?"
        else:
            # Last resort: try one more time with just the most important keywords
            keywords = extract_keywords_from_conversation(messages, query)
            if keywords:
                last_attempt_query = " ".join(keywords[:3])  # Just top 3 keywords
                last_context, last_scores = retrieve_context_from_faiss(doc_ids, last_attempt_query, top_k=5)
                
                if last_scores and max(last_scores.values()) > 0.15:
                    context_text = last_context
                    doc_score = last_scores
                else:
                    reply = "No relevant information found in the selected documents for your query."
            else:
                reply = "No relevant information found in the selected documents for your query."
    
    # Debug: Print context quality for troubleshooting
    print(f"DEBUG - Query: {query}")
    print(f"DEBUG - Doc scores: {doc_score}")
    print(f"DEBUG - Context length: {len(context_text) if context_text else 0}")
    print(f"DEBUG - Max score: {max(doc_score.values()) if doc_score else 0}")
    print(f"DEBUG - Context preview: {context_text[:200] if context_text else 'No context'}...")
    
    # Always try to use context if we have ANY reasonable score
    if doc_score and max(doc_score.values()) > 0.1 and context_text and len(context_text) > 50:
        # Force the system to use the context even if it seems marginal
        pass  # Continue to LLM processing
    elif 'reply' not in locals():
        # Only set reply if we haven't already set it above
        reply = "No relevant information found in the selected documents for your query."
    
    # Only proceed with LLM if we have some context
    if 'reply' not in locals():
        # Get top-ranked documents for the system prompt
        top_docs = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)[:3]
        doc_names_list = []
        documents = db["documents"]
        for doc_id, score in top_docs:
            doc_info = documents.find_one({"_id": doc_id}, {"name": 1})
            if doc_info:
                doc_names_list.append(f"{doc_info['name']} (relevance: {score:.2f})")
        
        # If we have any ranked documents, force the LLM to use them
        if doc_score and max(doc_score.values()) > 0.1:  # Very low threshold
            # Create system prompt with current context
            system_prompt = (
                "You are a helpful document context assistant. You MUST answer using the information provided in the context below. "
                "The context contains excerpts from specific documents with their names clearly marked. "
                f"TOP RANKED DOCUMENTS for this query: {', '.join(doc_names_list)}\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. MANDATORY: You have been provided with relevant document context below. You MUST use this information to answer the user's question.\n"
                "2. PRIORITY: Focus on information from the top-ranked documents listed above\n"
                "3. When referencing information, mention which document it came from\n"
                "4. If the answer requires combining information from multiple documents, do so\n"
                "5. Reference previous parts of our conversation when relevant\n"
                "6. Even if the information seems partial or indirect, use what's available in the context to provide a helpful response\n"
                "7. Only say 'No relevant information found' if the context is truly empty or completely unrelated\n"
                "8. Do NOT ignore information from highly ranked documents - they are ranked for a reason\n"
                "9. Try to extract useful information even if it's not a perfect match\n\n"
                "Context from selected documents:\n" + context_text
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
        else:
            reply = "No relevant information found in the selected documents for your query."

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
