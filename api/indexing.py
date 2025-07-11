import os
import subprocess
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
from pptx import Presentation
from nltk import sent_tokenize

load_dotenv()

router = APIRouter()

# ENV + DB
RESOURCE_DIR = Path("./resources")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]  # type: ignore
docs = db["documents"]

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def convert_ppt_to_pptx(file_path: Path) -> Path:
    """
    Converts a .ppt file to .pptx using LibreOffice CLI.
    Returns the new .pptx Path if successful, else raises Exception.
    """
    output_dir = file_path.parent
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to", "pptx",
        str(file_path),
        "--outdir", str(output_dir)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise Exception(f"LibreOffice conversion failed: {result.stderr.decode()}")
    new_file = output_dir / (file_path.stem + ".pptx")
    if not new_file.exists():
        raise Exception("Converted .pptx file not found after conversion.")
    return new_file

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
            ext = file_path.suffix.lower()
            text = ""

            if ext == ".pdf":
                doc = fitz.open(str(file_path))
                text = "\n".join(page.get_text() for page in doc)  # type: ignore

            elif ext in [".docx", ".doc"]:
                text = docx2txt.process(str(file_path))

            elif ext in [".ppt", ".pptx"]:
                # If .ppt, convert to .pptx first
                if ext == ".ppt":
                    file_path = convert_ppt_to_pptx(file_path)

                prs = Presentation(str(file_path))
                slides_text = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            slides_text.append(shape.text)
                text = "\n".join(slides_text)

            else:
                results.append({"id": doc_id, "status": f"unsupported_file: {ext}"})
                continue

            # Chunking & Embedding
            chunks = chunk_text(text)
            embeddings = [get_embedding(chunk) for chunk in chunks]
            embedding = np.array(embeddings, dtype="float32")

            # Create FAISS index
            dim = embedding.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embedding)  # type: ignore

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

# def chunk_text(text, max_tokens=1000):
#     sentences = text.split(". ")
#     chunks = []
#     current_chunk = ""

#     for sentence in sentences:
#         if len(current_chunk.split()) + len(sentence.split()) < max_tokens:
#             current_chunk += sentence + ". "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + ". "

#     if current_chunk:
#         chunks.append(current_chunk.strip())

#     return chunks

def chunk_text(text, max_tokens=500, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        tokens = sentence.split()
        if current_len + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            # add overlap
            current_chunk = current_chunk[-overlap:] + tokens
            current_len = len(current_chunk)
        else:
            current_chunk += tokens
            current_len += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks