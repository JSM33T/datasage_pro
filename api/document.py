import uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form,Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import shutil
import re

from fastapi import Body

load_dotenv()

router = APIRouter()

RESOURCE_DIR = Path("./resources")
RESOURCE_DIR.mkdir(exist_ok=True)

# Mongo setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")] # type: ignore
collection = db["documents"]

# Ensure text index exists
collection.create_index([
    ("name", "text"),
    ("description", "text"),
    ("generatedSummary", "text")
], name="DocumentTextIndex", default_language='english')

# Mongo setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")] # type: ignore
collection = db["documents"]

@router.get("/search")
def search_documents(query: str = Query(..., min_length=1)):
    # Case-insensitive regex search in name or filename first
    primary_results = list(collection.find(
        {
            "isIndexed": True,
            "$or": [
                {"name": {"$regex": re.escape(query), "$options": "i"}},
                {"filename": {"$regex": re.escape(query), "$options": "i"}}
            ]
        },
        {"_id": 1, "name": 1, "filename": 1, "isIndexed": 1, "description": 1, "generatedSummary": 1, "dateAdded": 1}
    ))

    # Convert ObjectId to string for all results
    def process_docs(docs):
        for doc in docs:
            doc["id"] = str(doc.get("_id", doc.get("id")))
            doc.pop("_id", None)
        return docs

    if primary_results:
        return process_docs(primary_results)

    secondary_results = list(collection.find(
        {
            "isIndexed": True,
            "description": {"$regex": re.escape(query), "$options": "i"}
        },
        {"_id": 1, "name": 1, "filename": 1, "isIndexed": 1, "description": 1, "generatedSummary": 1, "dateAdded": 1}
    ))

    return process_docs(secondary_results)

@router.get("/by_ids")
def get_documents_by_ids(ids: str = Query(..., description="Comma-separated document IDs")):
    id_list = ids.split(",")
    docs = list(collection.find(
        {"_id": {"$in": id_list}},
        {"_id": 1, "name": 1, "filename": 1, "isIndexed": 1, "description": 1, "generatedSummary": 1, "dateAdded": 1}
    ))
    for doc in docs:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return docs
 
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

        file_path = doc_dir / file.filename # type: ignore
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


@router.get("/list2")
def list_documents():
    docs = list(collection.find({}, {"_id": 1, "name": 1, "filename": 1, "isIndexed": 1, "description": 1, "generatedSummary": 1, "dateAdded": 1}))
    for doc in docs:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return docs

# @router.get("/list")
# def list_documents(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)):
#     skips = (page - 1) * page_size
#     cursor = collection.find(
#         {},
#         {"_id": 1, "name": 1, "filename": 1, "isIndexed": 1, "description": 1, "generatedSummary": 1, "dateAdded": 1}
#     ).skip(skips).limit(page_size)

#     docs = list(cursor)
#     for doc in docs:
#         doc["id"] = str(doc["_id"])
#         del doc["_id"]

#     total = collection.count_documents({})
#     return {"items": docs, "total": total, "page": page, "page_size": page_size}

@router.get("/listbak")
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = Query(None, description="Optional search by name")
):
    skips = (page - 1) * page_size

    query_filter = {}
    if search:
        query_filter["name"] = {"$regex": search, "$options": "i"}

    total = collection.count_documents(query_filter)
    cursor = collection.find(
        query_filter,
        {"_id": 1, "name": 1, "filename": 1, "isIndexed": 1, "description": 1, "generatedSummary": 1, "dateAdded": 1}
    ).skip(skips).limit(page_size)

    docs = list(cursor)
    for doc in docs:
        doc["id"] = str(doc["_id"])
        del doc["_id"]

    return {"items": docs, "total": total, "page": page, "page_size": page_size}

@router.get("/list")
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = Query(None, description="Optional search by name")
):
    skips = (page - 1) * page_size

    query_filter = {}
    if search:
        query_filter["name"] = {"$regex": search, "$options": "i"}

    total = collection.count_documents(query_filter)
    cursor = collection.find(
        query_filter,
        {"_id": 1, "name": 1, "filename": 1, "isIndexed": 1, "description": 1, "generatedSummary": 1, "dateAdded": 1}
    ).skip(skips).limit(page_size)

    docs = list(cursor)
    for idx, doc in enumerate(docs, start=skips + 1):
        doc["id"] = str(doc["_id"])
        doc["serial_no"] = idx
        del doc["_id"]

    return {"items": docs, "total": total, "page": page, "page_size": page_size}
