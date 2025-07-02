import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pymongo import MongoClient
import openai
import fitz
import docx2txt

load_dotenv()

router = APIRouter()

# ENV + DB
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
    return response.data[0].embedding

@router.post("/index")
async def index_document(data: dict = Body(...)):
    ids = data.get("doc_ids")
    if not ids or not isinstance(ids, list):
        return JSONResponse(status_code=400, content={"error": "doc_ids must be a list"})

    results = []
    for doc_id in ids:
        try:
            record = docs.find_one({"_id": doc_id})
            if not record:
                results.append({"id": doc_id, "status": "not_found"})
                continue

            doc_dir = RESOURCE_DIR / doc_id
            file_path = doc_dir / record["filename"]
            if not file_path.exists():
                results.append({"id": doc_id, "status": "file_missing"})
                continue

            # Extract text
            if file_path.suffix == ".pdf":
                doc = fitz.open(str(file_path))
                text = "\n".join(page.get_text() for page in doc) # type: ignore
            elif file_path.suffix == ".docx":
                text = docx2txt.process(str(file_path))
            else:
                results.append({"id": doc_id, "status": "unsupported_file"})
                continue

            # Get embedding
            #embedding = np.array([get_embedding(text)], dtype="float32")
            chunks = chunk_text(text)
            embeddings = [get_embedding(chunk) for chunk in chunks]
            embedding = np.array(embeddings, dtype="float32")
            
            dim = embedding.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embedding) # type: ignore

            # Save .faiss and .pkl
            faiss.write_index(index, str(doc_dir / f"{doc_id}.faiss"))
            with open(doc_dir / f"{doc_id}.pkl", "wb") as f:
                pickle.dump({"text": text, "filename": record["filename"]}, f)

            # Update DB
            docs.update_one({"_id": doc_id}, {"$set": {"isIndexed": True}})
            results.append({"id": doc_id, "status": "indexed"})

        except Exception as e:
            results.append({"id": doc_id, "status": f"error: {str(e)}"})

    return {"results": results}


def chunk_text(text, max_tokens=1000):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) < max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
