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

# LangChain imports for intelligent RAG
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any

# === Setup ===
load_dotenv(override=True)  # force reload
router = APIRouter()

RESOURCE_DIR = Path("./resources")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")] # type: ignore
sessions = db["chat_sessions"]
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize LangChain components
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Memory for conversation context
conversation_memory = ConversationBufferWindowMemory(
    k=5,  # Remember last 5 exchanges
    memory_key="chat_history",
    return_messages=True
)

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
    
# === LangChain-based Intelligent Document Processing ===

class IntelligentDocumentProcessor:
    """Advanced document processor using LangChain for understanding human intentions"""
    
    def __init__(self):
        self.llm = llm
        self.embeddings = embeddings
        self.intent_analyzer = self._create_intent_analyzer()
        self.document_analyzer = self._create_document_analyzer()
        self.query_enhancer = self._create_query_enhancer()
        
    def _create_intent_analyzer(self):
        """Create a chain to analyze user intent and query type"""
        intent_prompt = ChatPromptTemplate.from_template("""
        Analyze the user's query to understand their intent and information need.
        
        User Query: {query}
        Conversation History: {history}
        
        Classify the query into one of these categories:
        1. FACTUAL - Looking for specific facts, data, or information
        2. PROCEDURAL - Asking how to do something or about processes
        3. CONCEPTUAL - Understanding concepts, definitions, or explanations
        4. COMPARATIVE - Comparing different things or options
        5. TROUBLESHOOTING - Solving problems or issues
        6. EXPLORATORY - Open-ended exploration of a topic
        
        Also identify:
        - Key entities/topics mentioned
        - Specific domain or context
        - Level of detail required (brief, detailed, comprehensive)
        - Any implicit context from conversation history
        
        Provide your analysis in this format:
        Intent Category: [CATEGORY]
        Key Topics: [list of main topics]
        Context Domain: [domain/field]
        Detail Level: [brief/detailed/comprehensive]
        Enhanced Query: [reformulated query with better context]
        Search Keywords: [optimized keywords for document search]
        """)
        
        return LLMChain(llm=self.llm, prompt=intent_prompt)
    
    def _create_document_analyzer(self):
        """Create a chain to analyze document relevance and content matching"""
        doc_prompt = ChatPromptTemplate.from_template("""
        Analyze the following document content to determine its relevance to the user's query.
        
        User Query: {query}
        User Intent: {intent_analysis}
        Document Name: {doc_name}
        Document Content: {content}
        
        Evaluate:
        1. Relevance Score (0-100): How well does this document answer the query?
        2. Content Quality: Does the content directly address the user's intent?
        3. Information Completeness: How complete is the information for this query?
        4. Key Insights: What are the most relevant parts of this document?
        
        Provide analysis in this format:
        Relevance Score: [0-100]
        Quality Assessment: [high/medium/low]
        Completeness: [complete/partial/insufficient]
        Key Sections: [list of most relevant content sections]
        Confidence: [how confident you are in this assessment]
        """)
        
        return LLMChain(llm=self.llm, prompt=doc_prompt)
    
    def _create_query_enhancer(self):
        """Create a chain to enhance queries based on conversation context"""
        enhance_prompt = ChatPromptTemplate.from_template("""
        Enhance the user's query to better match relevant documents and improve search results.
        
        Original Query: {query}
        Intent Analysis: {intent}
        Conversation Context: {context}
        Document History: {doc_history}
        
        Create an enhanced query that:
        1. Includes implicit context from the conversation
        2. Uses domain-specific terminology when appropriate
        3. Clarifies ambiguous terms
        4. Adds relevant synonyms and related concepts
        5. Maintains the user's original intent
        
        Enhanced Query: [provide the improved query]
        Alternative Phrasings: [2-3 alternative ways to phrase the same query]
        """)
        
        return LLMChain(llm=self.llm, prompt=enhance_prompt)

# Initialize the intelligent processor
doc_processor = IntelligentDocumentProcessor()

def analyze_user_intent(query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """Analyze user intent and enhance query understanding"""
    try:
        history_text = ""
        if conversation_history:
            recent_messages = conversation_history[-6:]  # Last 3 exchanges
            history_text = "\n".join([
                f"User: {msg['content']}" if msg['role'] == 'user' 
                else f"Assistant: {msg['content']}" 
                for msg in recent_messages
            ])
        
        result = doc_processor.intent_analyzer.run(
            query=query,
            history=history_text
        )
        
        # Parse the result to extract structured information
        lines = result.strip().split('\n')
        analysis = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                analysis[key] = value.strip()
        
        return analysis
        
    except Exception as e:
        print(f"[DEBUG] Error in intent analysis: {e}")
        return {
            "intent_category": "FACTUAL",
            "key_topics": query,
            "enhanced_query": query,
            "search_keywords": query
        }

def create_intelligent_vectorstore(doc_ids: List[str]) -> FAISS:
    """Create a LangChain FAISS vectorstore from documents with metadata"""
    documents = []
    
    # Get document metadata
    doc_collection = db["documents"]
    doc_metadata = {}
    for doc in doc_collection.find({"_id": {"$in": doc_ids}}):
        doc_metadata[doc["_id"]] = {
            "name": doc.get("name", ""),
            "description": doc.get("description", ""),
            "filename": doc.get("filename", ""),
            "summary": doc.get("generatedSummary", "")
        }
    
    for doc_id in doc_ids:
        doc_dir = RESOURCE_DIR / doc_id
        pkl_path = doc_dir / f"{doc_id}.pkl"
        
        if not pkl_path.exists():
            continue
            
        try:
            with open(pkl_path, "rb") as f:
                meta = pickle.load(f)
                text_chunks = meta.get("text", [])
                if isinstance(text_chunks, str):
                    text_chunks = [text_chunks]
            
            doc_meta = doc_metadata.get(doc_id, {})
            
            for i, chunk in enumerate(text_chunks):
                if chunk.strip() and len(chunk.strip()) > 50:  # Filter out very short chunks
                    doc = Document(
                        page_content=chunk.strip(),
                        metadata={
                            "doc_id": doc_id,
                            "doc_name": doc_meta.get("name", f"Document_{doc_id}"),
                            "doc_description": doc_meta.get("description", ""),
                            "doc_summary": doc_meta.get("summary", ""),
                            "chunk_index": i,
                            "filename": doc_meta.get("filename", "")
                        }
                    )
                    documents.append(doc)
                    
        except Exception as e:
            print(f"[DEBUG] Error processing document {doc_id}: {e}")
            continue
    
    if not documents:
        return None
        
    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def intelligent_document_retrieval(
    query: str, 
    intent_analysis: Dict[str, Any], 
    vectorstore: FAISS, 
    k: int = 5
) -> List[Document]:
    """Retrieve documents using intelligent search based on intent analysis"""
    
    # Use enhanced query for better retrieval
    enhanced_query = intent_analysis.get("enhanced_query", query)
    search_keywords = intent_analysis.get("search_keywords", query)
    
    # Multi-strategy retrieval
    retrieved_docs = []
    
    try:
        # Strategy 1: Semantic similarity search with enhanced query
        similarity_docs = vectorstore.similarity_search(enhanced_query, k=k)
        retrieved_docs.extend(similarity_docs)
        
        # Strategy 2: Keyword-based search if different from enhanced query
        if search_keywords != enhanced_query:
            keyword_docs = vectorstore.similarity_search(search_keywords, k=k//2)
            retrieved_docs.extend(keyword_docs)
        
        # Strategy 3: MMR (Maximum Marginal Relevance) for diversity
        mmr_docs = vectorstore.max_marginal_relevance_search(
            enhanced_query, k=k//2, fetch_k=k*2
        )
        retrieved_docs.extend(mmr_docs)
        
        # Remove duplicates while preserving order
        seen_content = set()
        unique_docs = []
        for doc in retrieved_docs:
            content_hash = hash(doc.page_content[:200])  # Use first 200 chars as identifier
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
                
        return unique_docs[:k*2]  # Return up to k*2 documents for analysis
        
    except Exception as e:
        print(f"[DEBUG] Error in intelligent retrieval: {e}")
        # Fallback to simple similarity search
        return vectorstore.similarity_search(query, k=k)

def analyze_document_relevance(
    query: str, 
    intent_analysis: Dict[str, Any], 
    documents: List[Document]
) -> List[Dict[str, Any]]:
    """Analyze each document's relevance using LangChain"""
    
    analyzed_docs = []
    
    for doc in documents:
        try:
            # Analyze document relevance
            relevance_analysis = doc_processor.document_analyzer.run(
                query=query,
                intent_analysis=str(intent_analysis),
                doc_name=doc.metadata.get("doc_name", "Unknown"),
                content=doc.page_content[:2000]  # First 2000 chars for analysis
            )
            
            # Extract relevance score
            relevance_score = 50  # Default
            try:
                for line in relevance_analysis.split('\n'):
                    if 'relevance score:' in line.lower():
                        score_text = line.split(':')[1].strip()
                        relevance_score = int(''.join(filter(str.isdigit, score_text)))
                        break
            except:
                pass
            
            analyzed_docs.append({
                "document": doc,
                "relevance_score": relevance_score,
                "analysis": relevance_analysis,
                "doc_id": doc.metadata.get("doc_id"),
                "doc_name": doc.metadata.get("doc_name")
            })
            
        except Exception as e:
            print(f"[DEBUG] Error analyzing document relevance: {e}")
            # Fallback scoring
            analyzed_docs.append({
                "document": doc,
                "relevance_score": 30,
                "analysis": "Basic relevance assessment",
                "doc_id": doc.metadata.get("doc_id"),
                "doc_name": doc.metadata.get("doc_name")
            })
    
    # Sort by relevance score
    analyzed_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
    return analyzed_docs

def create_intelligent_response(
    query: str, 
    intent_analysis: Dict[str, Any], 
    relevant_documents: List[Dict[str, Any]], 
    conversation_history: List[Dict]
) -> str:
    """Generate intelligent response using LangChain with context awareness"""
    
    # Prepare context from top relevant documents
    context_docs = []
    doc_sources = []
    
    for doc_info in relevant_documents[:5]:  # Top 5 most relevant
        if doc_info["relevance_score"] > 20:  # Minimum relevance threshold
            doc = doc_info["document"]
            context_docs.append(doc.page_content)
            doc_sources.append(doc_info["doc_name"])
    
    if not context_docs:
        return "I don't have relevant information to answer your query based on the available documents."
    
    # Create context-aware response prompt
    response_prompt = ChatPromptTemplate.from_template("""
    You are an intelligent document assistant. Answer the user's query based on the provided context.
    
    User Query: {query}
    User Intent: {intent_category}
    Query Type: {detail_level}
    
    Context from Documents:
    {context}
    
    Conversation History:
    {history}
    
    Instructions:
    1. Provide a direct, helpful answer based on the context
    2. Match the detail level requested by the user
    3. If the context doesn't fully answer the query, explain what information is available
    4. Use clear, professional language
    5. Reference document sources when relevant
    6. Consider the conversation history for context
    
    Answer:
    """)
    
    # Prepare conversation history
    history_text = ""
    if conversation_history:
        recent_messages = conversation_history[-4:]  # Last 2 exchanges
        history_text = "\n".join([
            f"User: {msg['content']}" if msg['role'] == 'user' 
            else f"Assistant: {msg['content']}" 
            for msg in recent_messages
        ])
    
    # Generate response
    try:
        response_chain = LLMChain(llm=llm, prompt=response_prompt)
        response = response_chain.run(
            query=query,
            intent_category=intent_analysis.get("intent_category", "FACTUAL"),
            detail_level=intent_analysis.get("detail_level", "detailed"),
            context="\n\n".join(context_docs),
            history=history_text
        )
        
        # Add source information
        if doc_sources:
            unique_sources = list(set(doc_sources))
            if len(unique_sources) <= 3:
                response += f"\n\n*Sources: {', '.join(unique_sources)}*"
            else:
                response += f"\n\n*Sources: {', '.join(unique_sources[:3])} and {len(unique_sources)-3} more*"
        
        return response.strip()
        
    except Exception as e:
        print(f"[DEBUG] Error generating intelligent response: {e}")
        return "I encountered an error while processing your query. Please try again."
    
# === Helper Functions ===
def get_embedding(text: str):
    """Legacy embedding function for backward compatibility"""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array([response.data[0].embedding], dtype="float32")

# === Legacy Functions (kept for backward compatibility) ===
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

# === Helper functions for enhanced RAG operation ===
def calculate_document_rankings(doc_ids, document_context_history):
    """Calculate comprehensive document rankings with decay factor"""
    from datetime import datetime, timedelta
    
    rankings = {}
    current_time = datetime.utcnow()
    
    for doc_id in doc_ids:
        history = document_context_history.get(doc_id, {})
        
        # Base metrics
        success_count = history.get("success_count", 0)
        last_score = history.get("last_score", 0)
        avg_score = history.get("avg_score", 0)
        total_queries = history.get("total_queries", 0)
        last_used = history.get("last_used")
        
        # Calculate success rate
        success_rate = success_count / max(total_queries, 1)
        
        # Calculate recency factor (documents used recently get higher priority)
        recency_factor = 1.0
        if last_used:
            try:
                if isinstance(last_used, str):
                    last_used_dt = datetime.fromisoformat(last_used.replace('Z', '+00:00'))
                else:
                    last_used_dt = last_used
                
                days_since_use = (current_time - last_used_dt).days
                recency_factor = max(0.1, 1.0 - (days_since_use * 0.1))  # Decay over time
            except:
                recency_factor = 0.5  # Default if date parsing fails
        
        # Calculate composite score
        composite_score = (
            last_score * 0.4 +           # Recent performance
            avg_score * 0.3 +            # Historical average
            success_rate * 0.2 +         # Success rate
            recency_factor * 0.1         # Recency boost
        )
        
        rankings[doc_id] = {
            "composite_score": composite_score,
            "success_count": success_count,
            "success_rate": success_rate,
            "last_score": last_score,
            "avg_score": avg_score,
            "recency_factor": recency_factor
        }
    
    return rankings

def retrieve_single_document_context(doc_id, query, doc_names, top_k=3):
    """Retrieve context from a single document"""
    doc_dir = RESOURCE_DIR / doc_id
    faiss_path = doc_dir / f"{doc_id}.faiss"
    pkl_path = doc_dir / f"{doc_id}.pkl"

    if not faiss_path.exists() or not pkl_path.exists():
        return None, {}

    try:
        index = faiss.read_index(str(faiss_path))
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
            text_chunks = meta.get("text", [])
            if isinstance(text_chunks, str):
                text_chunks = [text_chunks]

        if not text_chunks:
            return None, {}

        query_vec = get_embedding(query)
        D, I = index.search(query_vec, top_k)

        matched_chunks = []
        doc_score = {}

        for i, idx in enumerate(I[0]):
            if 0 <= idx < len(text_chunks):
                doc_name = doc_names.get(doc_id, f"Document_{doc_id}")
                chunk_with_source = f"[From {doc_name}]:\n{text_chunks[idx]}"
                matched_chunks.append(chunk_with_source)
                distance = float(D[0][i])
                score = round(1 / (distance + 1e-6), 2)
                doc_score[doc_id] = doc_score.get(doc_id, 0.0) + score

        return "\n\n".join(matched_chunks), doc_score
    except Exception as e:
        print(f"[DEBUG] Error in single document retrieval: {e}")
        return None, {}

def create_intelligent_document_order(doc_ids, doc_rankings):
    """Create intelligent document ordering based on comprehensive ranking"""
    if not doc_rankings:
        return doc_ids
    
    # Separate documents into tiers based on performance
    tier1_docs = []  # High performers
    tier2_docs = []  # Medium performers
    tier3_docs = []  # Low/unknown performers
    
    for doc_id in doc_ids:
        ranking = doc_rankings.get(doc_id, {})
        composite_score = ranking.get("composite_score", 0)
        success_count = ranking.get("success_count", 0)
        
        if composite_score > 0.15 and success_count > 2:
            tier1_docs.append((doc_id, composite_score))
        elif composite_score > 0.05 or success_count > 0:
            tier2_docs.append((doc_id, composite_score))
        else:
            tier3_docs.append((doc_id, composite_score))
    
    # Sort each tier by composite score
    tier1_docs.sort(key=lambda x: x[1], reverse=True)
    tier2_docs.sort(key=lambda x: x[1], reverse=True)
    tier3_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Combine tiers
    final_order = (
        [doc_id for doc_id, _ in tier1_docs] +
        [doc_id for doc_id, _ in tier2_docs] +
        [doc_id for doc_id, _ in tier3_docs]
    )
    
    return final_order

def calculate_enhanced_score(doc_id, base_score, document_context_history, priority_weight):
    """Calculate enhanced score with historical data and priority weighting"""
    history = document_context_history.get(doc_id, {})
    
    # Base score adjustments
    enhanced_score = base_score
    
    # Historical performance boost
    success_count = history.get("success_count", 0)
    avg_score = history.get("avg_score", 0)
    
    if success_count > 0:
        # Boost based on historical success (gradual, not exponential)
        historical_boost = min(success_count * 0.05, 0.3)  # Max 30% boost
        enhanced_score += historical_boost
        
        # Additional boost if average score is high
        if avg_score > 0.1:
            avg_boost = min(avg_score * 0.2, 0.2)  # Max 20% boost
            enhanced_score += avg_boost
    
    # Apply priority weight (based on document order)
    enhanced_score *= priority_weight
    
    # Apply recency factor from rankings
    recency_factor = history.get("recency_factor", 1.0)
    enhanced_score *= recency_factor
    
    return round(enhanced_score, 3)

def enhance_query_with_context(query, document_context_history, messages):
    """Intelligently enhance query with document context and conversation history"""
    enhanced_query = query
    
    # Find top performing documents
    top_docs = []
    if document_context_history:
        for doc_id, history in document_context_history.items():
            success_rate = history.get("success_rate", 0)
            avg_score = history.get("avg_score", 0)
            success_count = history.get("success_count", 0)
            
            if success_count > 0 and avg_score > 0.08:  # Meaningful threshold
                top_docs.append((doc_id, success_rate, avg_score, success_count))
    
    # Sort by composite ranking
    top_docs.sort(key=lambda x: (x[1] * 0.5 + x[2] * 0.3 + min(x[3] * 0.1, 0.2)), reverse=True)
    
    # Analyze recent conversation for context
    recent_context = ""
    if len(messages) > 2:
        # Look at last few messages for context
        last_messages = messages[-4:]  # Last 4 messages
        for msg in last_messages:
            if msg["role"] == "user":
                recent_context += msg["content"] + " "
    
    # Enhance query based on top performing documents
    if top_docs:
        best_doc_id = top_docs[0][0]
        best_success_rate = top_docs[0][1]
        best_avg_score = top_docs[0][2]
        
        # Only enhance if there's strong confidence in the document
        if best_success_rate > 0.3 and best_avg_score > 0.1:
            # Get document name
            documents = db["documents"]
            doc_info = documents.find_one({"_id": best_doc_id}, {"name": 1})
            if doc_info:
                doc_name = doc_info["name"]
                
                # Context-aware enhancement
                if "what" in query.lower() or "how" in query.lower() or "explain" in query.lower():
                    enhanced_query = f"{query} (focusing on {doc_name})"
                elif len(recent_context) > 20:  # If there's conversation context
                    enhanced_query = f"{query} (in context of {doc_name})"
                else:
                    enhanced_query = f"{query} wrt({doc_name})"
                
                print(f"[DEBUG] Enhanced query with top document context: {enhanced_query}")
                print(f"[DEBUG] Top document: {doc_name} (success_rate: {best_success_rate:.2f}, avg_score: {best_avg_score:.3f})")
    
    return enhanced_query

def attempt_ai_processing_fallback(query, enhanced_query, doc_score, document_context_history):
    """Attempt AI processing fallback when standard RAG fails"""
    # Check if we have any document with score > 5% for AI processing
    if doc_score and max(doc_score.values()) > 0.05:
        max_score = max(doc_score.values())
        top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
        
        print(f"[DEBUG] Low context but score {max_score:.3f} > 5%, trying AI processing")
        
        return process_document_with_ai(top_doc_id, enhanced_query, max_score)
    else:
        return "I don't know."

def process_with_standard_rag(context_text_final, messages, query, enhanced_query, doc_score, document_context_history):
    """Process query with standard RAG approach"""
    system_prompt = (
        "You are a helpful document context assistant. Answer ONLY using the information in the provided context below. "
        "If the answer is not present in the context, reply with 'No relevant document found for your query.'\n\nContext:\n" + context_text_final
    )
    
    try:
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
        reply = gpt_response.choices[0].message.content.strip()
        
        # Post-processing check: If LLM still refuses despite high score, force AI processing
        if (doc_score and max(doc_score.values()) > 0.05 and 
            ("no relevant" in reply.lower() or "not found" in reply.lower() or "don't have" in reply.lower())):
            
            max_score = max(doc_score.values())
            top_doc_id = max(doc_score.items(), key=lambda x: x[1])[0]
            
            print(f"[DEBUG] LLM refused despite score {max_score:.3f}, forcing AI processing")
            
            ai_reply = process_document_with_ai(top_doc_id, enhanced_query, max_score)
            if ai_reply and ai_reply != "I don't know.":
                return ai_reply
        
        return reply
    except Exception as e:
        print(f"[DEBUG] Error in standard RAG processing: {e}")
        return "I encountered an error while processing your query."

def process_document_with_ai(doc_id, query, score):
    """Process a document using AI when standard RAG fails"""
    doc_dir = RESOURCE_DIR / doc_id
    pkl_path = doc_dir / f"{doc_id}.pkl"
    
    if not pkl_path.exists():
        return "I don't know."
    
    try:
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
            text_chunks = meta.get("text", [])
            if isinstance(text_chunks, str):
                text_chunks = [text_chunks]
        
        if not text_chunks:
            return "I don't know."
        
        # Get meaningful chunks with better selection
        meaningful_chunks = []
        for chunk in text_chunks[:15]:  # Increased chunk count
            if chunk.strip() and len(chunk.strip()) > 30:
                meaningful_chunks.append(chunk.strip())
            if len(meaningful_chunks) >= 5:  # Get more chunks for better context
                break
        
        if not meaningful_chunks:
            return "I don't know."
        
        # Get document name
        documents = db["documents"]
        doc_info = documents.find_one({"_id": doc_id}, {"name": 1})
        doc_name = doc_info["name"] if doc_info else f"Document_{doc_id}"
        
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
        reply = f"{processed_reply}\n\n*Source: {doc_name} (relevance score: {score:.1f}%)*"
        
        print(f"[DEBUG] AI processing successful from {doc_name}")
        return reply
        
    except Exception as e:
        print(f"[DEBUG] Error in AI processing: {e}")
        return "I don't know."

# === Enhanced context retrieval with conversation history priority ===
def retrieve_context_with_priority(doc_ids, query, document_context_history, top_k=5):
    """Retrieve context prioritizing documents that have been successful in conversation history"""
    
    # Get document names for better context
    documents = db["documents"]
    doc_cursor = documents.find({"_id": {"$in": doc_ids}}, {"_id": 1, "name": 1})
    doc_names = {doc["_id"]: doc["name"] for doc in doc_cursor}
    
    # Calculate comprehensive document scores including historical performance
    doc_rankings = calculate_document_rankings(doc_ids, document_context_history)
    
    # Find the most successful document from history with decay factor
    best_doc_id = None
    best_composite_score = 0
    
    if document_context_history and doc_rankings:
        for doc_id, ranking in doc_rankings.items():
            if (ranking["composite_score"] > best_composite_score and 
                ranking["composite_score"] > 0.1):  # Minimum threshold
                best_doc_id = doc_id
                best_composite_score = ranking["composite_score"]
    
    # First, try to get context from the best document only if it's significantly better
    if best_doc_id and best_doc_id in doc_ids and best_composite_score > 0.15:
        print(f"[DEBUG] First checking best document: {doc_names.get(best_doc_id, best_doc_id)} (composite: {best_composite_score:.3f})")
        
        single_doc_context, single_doc_score = retrieve_single_document_context(
            best_doc_id, query, doc_names, top_k
        )
        
        if single_doc_context and single_doc_score:
            max_score = max(single_doc_score.values())
            # Use adaptive threshold based on historical performance
            threshold = max(0.08, best_composite_score * 0.5)
            
            if max_score > threshold:
                print(f"[DEBUG] Best document provides good results: {max_score:.3f} > {threshold:.3f}")
                return single_doc_context, single_doc_score
            else:
                print(f"[DEBUG] Best document score too low: {max_score:.3f} <= {threshold:.3f}, checking other documents")
    
    # If best document doesn't provide good results, check all documents with smart ordering
    print(f"[DEBUG] Checking all documents with intelligent prioritization")
    
    # Create intelligent document ordering based on comprehensive ranking
    final_doc_order = create_intelligent_document_order(doc_ids, doc_rankings)
    
    print(f"[DEBUG] Document processing order: {[doc_names.get(doc_id, doc_id[:8]) for doc_id in final_doc_order[:5]]} (showing top 5)")
    
    # Try to get context from all documents with weighted scoring
    combined_index = None
    all_text_chunks = []
    chunk_doc_map = []
    doc_priority_weights = {}
    
    for i, doc_id in enumerate(final_doc_order):
        # Calculate priority weight (higher for earlier documents)
        priority_weight = 1.0 / (i + 1) if i < 10 else 0.1
        doc_priority_weights[doc_id] = priority_weight
        
        doc_dir = RESOURCE_DIR / doc_id
        faiss_path = doc_dir / f"{doc_id}.faiss"
        pkl_path = doc_dir / f"{doc_id}.pkl"

        if not faiss_path.exists() or not pkl_path.exists():
            continue

        try:
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
        except Exception as e:
            print(f"[DEBUG] Error loading document {doc_id}: {e}")
            continue

    if combined_index is None or not all_text_chunks:
        return "No relevant document content found.", {}

    try:
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
                base_score = round(1 / (distance + 1e-6), 2)
                
                # Apply sophisticated scoring with historical data
                final_score = calculate_enhanced_score(
                    doc_id, base_score, document_context_history, 
                    doc_priority_weights.get(doc_id, 1.0)
                )
                
                doc_score[doc_id] = doc_score.get(doc_id, 0.0) + final_score

        return "\n\n".join(matched_chunks), doc_score
    except Exception as e:
        print(f"[DEBUG] Error in context retrieval: {e}")
        return "Error retrieving context.", {}


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
    """Enhanced chat continuation using LangChain for intelligent document understanding"""
    user_id = request.state.user.get('username')
    
    session_id = data.get("session_id")
    query = data.get("query")
    if not session_id or not query:
        return JSONResponse(status_code=400, content={"error": "session_id and query required"})

    # Find session by ID and ensure it belongs to the user
    chat = sessions.find_one({"_id": session_id, "user_id": user_id})
    if not chat:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    messages = chat["messages"]
    
    print(f"\n[DEBUG] === LangChain-Enhanced RAG Processing ===")
    print(f"[DEBUG] User Query: {query}")
    
    # Step 1: Analyze user intent and enhance query
    intent_analysis = analyze_user_intent(query, messages)
    print(f"[DEBUG] Intent Analysis: {intent_analysis}")
    
    # Step 2: Get all available documents
    all_doc_ids = [doc["_id"] for doc in db["documents"].find({}, {"_id": 1})]
    
    # Step 3: Create intelligent vectorstore
    print(f"[DEBUG] Creating vectorstore from {len(all_doc_ids)} documents...")
    vectorstore = create_intelligent_vectorstore(all_doc_ids)
    
    if not vectorstore:
        return JSONResponse(
            status_code=500, 
            content={"error": "No documents available for processing"}
        )
    
    # Step 4: Intelligent document retrieval
    print(f"[DEBUG] Performing intelligent document retrieval...")
    retrieved_docs = intelligent_document_retrieval(
        query, intent_analysis, vectorstore, k=8
    )
    
    print(f"[DEBUG] Retrieved {len(retrieved_docs)} document chunks")
    
    # Step 5: Analyze document relevance
    print(f"[DEBUG] Analyzing document relevance...")
    analyzed_docs = analyze_document_relevance(query, intent_analysis, retrieved_docs)
    
    # Filter for high relevance documents
    relevant_docs = [doc for doc in analyzed_docs if doc["relevance_score"] > 25]
    print(f"[DEBUG] Found {len(relevant_docs)} highly relevant documents")
    
    # Step 6: Generate intelligent response
    print(f"[DEBUG] Generating intelligent response...")
    
    # Add user message to conversation
    messages.append({"role": "user", "content": query})
    
    # Generate response using LangChain
    if relevant_docs:
        reply = create_intelligent_response(
            query, intent_analysis, relevant_docs, messages
        )
    else:
        reply = "I couldn't find relevant information in the available documents to answer your query. Could you please rephrase your question or provide more specific details?"
    
    # Add assistant response to conversation
    messages.append({"role": "assistant", "content": reply})
    
    # Step 7: Update conversation history and document tracking
    document_context_history = chat.get("document_context_history", {})
    doc_score = {}
    
    # Update document scores based on relevance analysis
    for doc_info in relevant_docs:
        doc_id = doc_info["doc_id"]
        relevance_score = doc_info["relevance_score"] / 100.0  # Normalize to 0-1
        doc_score[doc_id] = relevance_score
        
        # Update context history
        if doc_id not in document_context_history:
            document_context_history[doc_id] = {
                "success_count": 0,
                "total_queries": 0,
                "last_score": 0,
                "avg_score": 0,
                "score_history": [],
                "last_used": None
            }
        
        history = document_context_history[doc_id]
        history["total_queries"] += 1
        history["last_score"] = relevance_score
        history["last_used"] = datetime.utcnow().isoformat()
        
        # Update score history
        score_history = history.get("score_history", [])
        score_history.append(relevance_score)
        if len(score_history) > 10:
            score_history = score_history[-10:]
        history["score_history"] = score_history
        
        # Update average score
        history["avg_score"] = sum(score_history) / len(score_history)
        
        # Update success count (relevance > 0.3 considered success)
        if relevance_score > 0.3:
            history["success_count"] += 1
        
        # Calculate success rate
        history["success_rate"] = history["success_count"] / history["total_queries"]
    
    # Limit message history
    MAX_MESSAGES = 12
    messages = messages[-MAX_MESSAGES:]
    
    # Save updated session
    sessions.update_one(
        {"_id": session_id, "user_id": user_id},
        {"$set": {
            "messages": messages,
            "document_context_history": document_context_history,
            "last_intent_analysis": intent_analysis
        }}
    )
    
    # Prepare matched documents for response
    matched_docs = []
    for doc_info in relevant_docs[:5]:  # Top 5 documents
        doc_id = doc_info["doc_id"]
        doc_name = doc_info["doc_name"]
        
        # Get document metadata
        doc_meta = db["documents"].find_one({"_id": doc_id}, {"filename": 1})
        filename = doc_meta.get("filename", "") if doc_meta else ""
        
        matched_docs.append({
            "doc_id": str(doc_id),
            "doc_name": doc_name,
            "link": f"/resources/{doc_id}/{filename}",
            "score": round(doc_info["relevance_score"], 1)
        })
    
    print(f"[DEBUG] Response generated successfully")
    print(f"[DEBUG] Matched {len(matched_docs)} documents")
    print(f"[DEBUG] === End LangChain Processing ===\n")
    
    return {
        "reply": reply,
        "messages": messages,
        "matched_docs": matched_docs,
        "intent_analysis": intent_analysis
    }

@router.post("/test_langchain")
def test_langchain_functionality(data: dict = Body(...), request: Request = None):
    """Test endpoint for LangChain-enhanced RAG functionality"""
    user_id = request.state.user.get('username')
    query = data.get("query", "What is this document about?")
    
    try:
        # Get a few documents for testing
        all_doc_ids = [doc["_id"] for doc in db["documents"].find({}, {"_id": 1}).limit(5)]
        
        if not all_doc_ids:
            return {"error": "No documents found for testing"}
        
        # Step 1: Analyze intent
        intent_analysis = analyze_user_intent(query, [])
        
        # Step 2: Create vectorstore
        vectorstore = create_intelligent_vectorstore(all_doc_ids)
        
        if not vectorstore:
            return {"error": "Could not create vectorstore"}
        
        # Step 3: Retrieve documents
        retrieved_docs = intelligent_document_retrieval(query, intent_analysis, vectorstore, k=3)
        
        # Step 4: Analyze relevance
        analyzed_docs = analyze_document_relevance(query, intent_analysis, retrieved_docs)
        
        # Step 5: Generate response
        relevant_docs = [doc for doc in analyzed_docs if doc["relevance_score"] > 20]
        
        if relevant_docs:
            response = create_intelligent_response(query, intent_analysis, relevant_docs, [])
        else:
            response = "No relevant documents found for the test query."
        
        return {
            "query": query,
            "intent_analysis": intent_analysis,
            "documents_analyzed": len(analyzed_docs),
            "relevant_documents": len(relevant_docs),
            "response": response,
            "top_docs": [
                {
                    "doc_name": doc["doc_name"],
                    "relevance_score": doc["relevance_score"]
                } for doc in relevant_docs[:3]
            ]
        }
        
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}


