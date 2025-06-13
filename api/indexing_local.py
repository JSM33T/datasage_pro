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
import fitz
import docx2txt
from sentence_transformers import SentenceTransformer

# === ENV & DB =================================================================
load_dotenv()
RESOURCE_DIR = Path("./resources")
client      = MongoClient(os.getenv("MONGO_URI"))
db          = client[os.getenv("MONGO_DB")]
docs        = db["documents"]

# === Model =====================================================================
model = SentenceTransformer("all-MiniLM-L6-v2")      # 384‑dim embeddings


# === Helpers ===================================================================
def chunk_text(text: str, size: int = 1024, overlap: int = 128):
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step)]

def embed_chunks_BIG(chunks: list[str]) -> np.ndarray:
    return np.asarray(model.encode(chunks, batch_size=32, show_progress_bar=False), dtype="float32")

def embed_chunks(chunks: list[str]) -> np.ndarray:
    prompts = [f"query: {chunk}" for chunk in chunks]
    return np.asarray(model.encode(prompts, batch_size=32, show_progress_bar=False), dtype="float32")

# === API =======================================================================
router = APIRouter()

@router.post("/index")
async def index_document(data: dict = Body(...)):
    ids = data.get("doc_ids")
    if not ids or not isinstance(ids, list):
        return JSONResponse(400, {"error": "doc_ids must be a list"})

    results: list[dict] = []

    for doc_id in ids:
        try:
            rec = docs.find_one({"_id": doc_id})
            if not rec:
                results.append({"id": doc_id, "status": "not_found"})
                continue

            doc_dir   = RESOURCE_DIR / doc_id
            file_path = doc_dir / rec["filename"]
            if not file_path.exists():
                results.append({"id": doc_id, "status": "file_missing"})
                continue

            # --- Extract text --------------------------------------------------
            if file_path.suffix.lower() == ".pdf":
                pdf  = fitz.open(file_path)
                text = "\n".join(p.get_text() for p in pdf)
            elif file_path.suffix.lower() == ".docx":
                text = docx2txt.process(file_path)
            else:
                results.append({"id": doc_id, "status": "unsupported_file"})
                continue

            chunks   = chunk_text(text)
            vectors  = embed_chunks(chunks)                   # (n,384)
            dim      = vectors.shape[1]
            index    = faiss.IndexFlatL2(dim)
            index.add(vectors)

            # --- Persist -------------------------------------------------------
            faiss.write_index(index, str(doc_dir / f"{doc_id}.faiss"))
            with open(doc_dir / f"{doc_id}.pkl", "wb") as fp:
                pickle.dump(
                    {
                        "text": chunks,
                        "filename": rec["filename"],
                        "dim": dim,
                        "createdAt": datetime.utcnow()
                    },
                    fp,
                )

            docs.update_one({"_id": doc_id}, {"$set": {"isIndexed": True}})
            results.append({"id": doc_id, "status": "indexed"})

        except Exception as e:
            results.append({"id": doc_id, "status": f"error: {e}"})

    return {"results": results}
