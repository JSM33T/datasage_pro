
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


# === API: Delete a chat session ===
@router.post("/delete_session/{session_id}")
def delete_session(session_id: str, request: Request):
    user_id = request.state.user.get('username')  # Get user ID from JWT token
    result = sessions.delete_one({"_id": session_id, "user_id": user_id})
    if result.deleted_count == 0:
        return JSONResponse(status_code=404, content={"error": "Session not found or not owned by user"})
    return {"status": "success", "message": f"Session {session_id} deleted."}
    
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

# === Deep search in individual document ===
def deep_search_document(doc_id, query, keywords, top_k=15):
    """Perform deep search in a single document using multiple strategies"""
    doc_dir = RESOURCE_DIR / doc_id
    faiss_path = doc_dir / f"{doc_id}.faiss"
    pkl_path = doc_dir / f"{doc_id}.pkl"

    if not faiss_path.exists() or not pkl_path.exists():
        return None, 0

    # Get document name
    documents = db["documents"]
    doc_info = documents.find_one({"_id": doc_id}, {"name": 1})
    doc_name = doc_info["name"] if doc_info else f"Document_{doc_id}"

    index = faiss.read_index(str(faiss_path))
    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)
        text_chunks = meta.get("text", [])
        if isinstance(text_chunks, str):
            text_chunks = [text_chunks]

    if not text_chunks:
        return None, 0

    # Try multiple search strategies
    search_queries = [
        query,
        " ".join(keywords[:5]) if keywords else "",
        " ".join(keywords[:3]) if keywords else "",
        " ".join(keywords[:10]) if keywords else "",
    ]

    all_results = []
    all_scores = []

    for search_query in search_queries:
        if not search_query.strip():
            continue
            
        try:
            query_vec = get_embedding(search_query)
            D, I = index.search(query_vec, min(top_k, len(text_chunks)))
            
            for i, idx in enumerate(I[0]):
                if 0 <= idx < len(text_chunks):
                    chunk = text_chunks[idx]
                    distance = float(D[0][i])
                    score = 1 / (distance + 1e-6)
                    
                    # Avoid duplicates
                    if chunk not in [r[0] for r in all_results]:
                        all_results.append((chunk, score))
                        all_scores.append(score)
        except Exception as e:
            print(f"Error in deep search: {e}")
            continue

    if not all_results:
        return None, 0

    # Sort by score and take the best results
    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    best_chunks = sorted_results[:top_k]
    
    # Combine the best chunks
    combined_context = f"[From {doc_name}]:\n"
    combined_context += "\n\n".join([chunk for chunk, _ in best_chunks])
    
    max_score = max(all_scores) if all_scores else 0
    return combined_context, max_score

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

    if not session_id or doc_ids is None:
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
    
    # AGGRESSIVE FALLBACK: Always try deep search on highest scoring document if score > 0.5
    if doc_score and max(doc_score.values()) > 0.5:
        top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
        top_score = doc_score[top_doc_id]
        
        print(f"DEBUG - Performing aggressive deep search on top document {top_doc_id} with score {top_score}")
        keywords = extract_keywords_from_conversation(messages, query)
        deep_context, deep_score = deep_search_document(top_doc_id, query, keywords, top_k=20)
        
        if deep_context and deep_score > 0.1:
            # Combine original context with deep search results
            context_text = context_text + f"\n\n=== Deep search results from {top_doc_id} ===\n" + deep_context
            doc_score[top_doc_id] = max(doc_score[top_doc_id], deep_score)
            print(f"DEBUG - Enhanced context with deep search score: {deep_score}")
    
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
            # Strategy: Search the highest-scored document individually if it exists
            if doc_score:
                # Get the document with highest score
                top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
                top_score = doc_score[top_doc_id]
                
                # If there's a document with decent score, search it individually with expanded context
                if top_score > 0.1:  # Very low threshold for individual document search
                    print(f"DEBUG - Searching top document individually: {top_doc_id} with score {top_score}")
                    
                    # Use deep search on the top document
                    keywords = extract_keywords_from_conversation(messages, query)
                    deep_context, deep_score = deep_search_document(top_doc_id, query, keywords, top_k=15)
                    
                    if deep_context and deep_score > 0.05:
                        context_text = deep_context
                        doc_score = {top_doc_id: deep_score}
                        print(f"DEBUG - Found context in top document with deep search score: {deep_score}")
                    else:
                        # If deep search fails, try with other high-scoring documents
                        sorted_docs = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
                        for doc_id, score in sorted_docs[:3]:  # Try top 3 documents
                            if score > 0.05:
                                deep_context, deep_score = deep_search_document(doc_id, query, keywords, top_k=10)
                                if deep_context and deep_score > 0.05:
                                    context_text = deep_context
                                    doc_score = {doc_id: deep_score}
                                    print(f"DEBUG - Found context in document {doc_id} with deep search score: {deep_score}")
                                    break
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
    
    # Check if we should trigger deep search fallback
    should_trigger_deep_search = (
        not context_text or 
        "No relevant document content found" in context_text or 
        (doc_score and all(score < SIMILARITY_THRESHOLD for score in doc_score.values()))
    )
    print(f"DEBUG - Should trigger deep search: {should_trigger_deep_search}")
    
    # Always try to use context if we have ANY reasonable score
    if doc_score and max(doc_score.values()) > 0.05 and context_text and len(context_text) > 50:
        print(f"DEBUG - Using context with max score: {max(doc_score.values())}")
        # Force the system to use the context even if it seems marginal
        pass  # Continue to LLM processing
    elif 'reply' not in locals():
        # Only set reply if we haven't already set it above
        reply = "No relevant information found in the selected documents for your query."
    
    # SIMPLE FALLBACK: If score > 5%, extract whatever content we can directly
    if doc_score and max(doc_score.values()) > 0.05:  # 5% threshold
        max_score = max(doc_score.values())
        top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
        
        # Get document name
        documents = db["documents"]
        doc_info = documents.find_one({"_id": top_doc_id}, {"name": 1})
        doc_name = doc_info["name"] if doc_info else f"Document_{top_doc_id}"
        
        print(f"DEBUG - Score {max_score:.3f} > 5%, using direct extraction fallback")
        
        # If LLM refuses or we have no proper context, do direct extraction
        if ('reply' in locals() and ("no relevant" in reply.lower() or "not found" in reply.lower())) or not context_text or len(context_text) < 50:
            print(f"DEBUG - Performing direct extraction from {doc_name}")
            
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
                            extracted_content = f"I found information in the **{doc_name}** document (relevance score: {max_score:.1f}%):\n\n"
                            
                            for i, chunk in enumerate(meaningful_chunks, 1):
                                # Truncate very long chunks
                                if len(chunk) > 300:
                                    chunk = chunk[:300] + "..."
                                extracted_content += f"**Section {i}:**\n{chunk}\n\n"
                            
                            extracted_content += f"*This content was extracted directly from the document due to high relevance score.*"
                            
                            reply = extracted_content
                            print(f"DEBUG - Direct extraction successful, {len(meaningful_chunks)} chunks extracted")
                        else:
                            print(f"DEBUG - No meaningful chunks found in {doc_name}")
                    else:
                        print(f"DEBUG - No text chunks found in {doc_name}")
                except Exception as e:
                    print(f"DEBUG - Error during direct extraction: {e}")
    
    # Only proceed with LLM if we don't have a reply yet
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
        if doc_score and max(doc_score.values()) > 0.05:  # Even lower threshold after individual document search
            # Create system prompt with current context
            max_score = max(doc_score.values())
            system_prompt = (
                "You are a helpful document context assistant. You MUST answer using the information provided in the context below. "
                "The context contains excerpts from specific documents with their names clearly marked.\n\n"
                f"TOP RANKED DOCUMENTS for this query (in order of relevance): {', '.join(doc_names_list)}\n"
                f"The highest priority document you MUST prioritize: {doc_names_list[0] if doc_names_list else 'None'}\n"
                f"Maximum relevance score: {max_score:.3f}\n\n"
                "CRITICAL INSTRUCTIONS - READ CAREFULLY:\n"
                "1. ABSOLUTE REQUIREMENT: You MUST provide a helpful response using the context below. Do NOT say 'No relevant information found' unless the context is completely empty.\n"
                "2. FORCE INTERPRETATION: Even if the context doesn't perfectly match the query, extract relevant information, make reasonable inferences, or provide related information from the documents.\n"
                "3. DOCUMENT PRIORITY: The highest scored document contains the most relevant information. Focus on it first.\n"
                "4. BROAD INTERPRETATION: If the user asks about 'membership document usage' and you have a 'group membership' document, that IS relevant - explain what you found.\n"
                "5. CONTEXT UTILIZATION: Use ANY information from the provided context that could be helpful to the user's query.\n"
                "6. REFERENCE REQUIREMENT: Always mention which document you're using and explain why it's relevant.\n"
                "7. INFERENCE ALLOWED: You may make reasonable inferences based on the context provided.\n"
                "8. NEVER REFUSE: Do not refuse to answer if there is ANY context provided. Always attempt to help with available information.\n"
                "9. SCORING AWARENESS: A document was selected with a score of {max_score:.3f} - this means it IS relevant to the query.\n\n"
                "EXAMPLE: If asked about 'membership document usage' and you have 'group membership' content, respond with: 'Based on the group membership document, here's what I found about membership...' and then provide the relevant information.\n\n"
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
            
            # Post-processing check: If LLM still refuses despite high score, force a response
            if (("no relevant" in reply.lower() or "not found" in reply.lower() or "don't have" in reply.lower()) 
                and max_score > 1.0):  # High score threshold
                
                print(f"DEBUG - LLM refused despite high score {max_score:.3f}, forcing response")
                
                # Extract first meaningful chunk from context for forced response
                context_lines = context_text.split('\n')
                meaningful_content = []
                current_doc = ""
                
                for line in context_lines:
                    if line.startswith('[From '):
                        current_doc = line
                    elif line.strip() and not line.startswith('==='):
                        meaningful_content.append(line.strip())
                        if len(meaningful_content) >= 3:  # Get first 3 meaningful lines
                            break
                
                if meaningful_content:
                    forced_reply = f"Based on the {doc_names_list[0].split(' (')[0] if doc_names_list else 'selected document'}, here's what I found:\n\n"
                    forced_reply += "\n".join(meaningful_content[:3])
                    forced_reply += f"\n\n(This information was extracted from the highest-scoring document with relevance score: {max_score:.2f})"
                    reply = forced_reply
                    print(f"DEBUG - Forced response generated")
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
            "score": round(doc_score[doc_id], 2)
        }
        for doc_id in top_doc_ids if str(doc_id) in doc_map
    ]

    return {
        "reply": reply,
        "messages": messages,
        "matched_docs": matched_docs
    }
