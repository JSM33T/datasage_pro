import uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import shutil
from fastapi import Body

load_dotenv()

router = APIRouter()

RESOURCE_DIR = Path("./resources")
RESOURCE_DIR.mkdir(exist_ok=True)

# Mongo setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]
collection = db["documents"]

@router.post("/add")
async def add_document(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(...)
):
    try:
        # ✅ Check for duplicate
        if collection.find_one({"filename": file.filename}):
            return JSONResponse(status_code=400, content={"error": "File with this name already exists"})

        doc_id = str(uuid.uuid4())
        doc_dir = RESOURCE_DIR / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        file_path = doc_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        collection.insert_one({
            "_id": doc_id,
            "name": name,
            "filename": file.filename,
            "description": description,
            "generatedSummary": "",
            "isIndexed": False,
            "dateAdded": datetime.utcnow()
        })

        return {
            "id": doc_id,
            "filename": file.filename,
            "name": name,
            "isIndexed": False,
            "description": description,
            "generatedSummary": ""
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/delete")
def delete_doc(data: dict = Body(...)):
    doc_id = data.get("doc_id")
    if not doc_id:
        return JSONResponse(status_code=400, content={"error": "Missing doc_id"})

    doc_dir = RESOURCE_DIR / doc_id
    if not doc_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Document not found"})

    # Optional: remove from Mongo
    collection.delete_one({"_id": doc_id})

    # Delete folder
    shutil.rmtree(doc_dir, ignore_errors=True)
    return {"status": "deleted", "id": doc_id}


@router.get("/list")
def list_documents():
    docs = list(collection.find({}, {"_id": 1, "name": 1, "filename": 1, "isIndexed": 1, "description": 1, "generatedSummary": 1, "dateAdded": 1}))
    for doc in docs:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return docs