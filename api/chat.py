from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pymongo import MongoClient
from pathlib import Path
import faiss
import pickle
import numpy as np
import openai
import os

load_dotenv()

router = APIRouter()

RESOURCE_DIR = Path("./resources")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")] # type: ignore
docs = db["documents"]

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array([response.data[0].embedding], dtype="float32")

@router.post("/chat")
async def chat_with_docs(data: dict = Body(...)):
    doc_ids = data.get("doc_ids")
    query = data.get("query")
    if not doc_ids or not query:
        return JSONResponse(status_code=400, content={"error": "Missing doc_ids or query"})

    try:
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

        if combined_index is None:
            return JSONResponse(status_code=404, content={"error": "No index files found"})

        query_vec = get_embedding(query)
        top_k = 3
        D, I = combined_index.search(query_vec, top_k)

        context = "\n\n".join(all_metadata[i].get("text", "") for i in I[0] if i < len(all_metadata))

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents.Be very precise and dont hit around the bush. reply to /json with the jsonified data that you can extract from the document"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        chat_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages # type: ignore
        )

        answer = chat_response.choices[0].message.content.strip() # type: ignore

        return {
            "query": query,
            "answer": answer,
            "context": context
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
