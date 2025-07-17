import uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Query, Request
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import shutil
import re
import pytz

from fastapi import Body

load_dotenv()

router = APIRouter()

RESOURCE_DIR = Path("./resources")
RESOURCE_DIR.mkdir(exist_ok=True)

# Helper function to get current Indian time
def get_indian_time():
    """Get current time in Indian Standard Time (IST) - timezone naive for MongoDB"""
    utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    ist = pytz.timezone('Asia/Kolkata')
    ist_time = utc_now.astimezone(ist)
    # Return timezone-naive datetime in IST for MongoDB compatibility
    return ist_time.replace(tzinfo=None)

# Mongo setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]  # type: ignore
collection = db["documents"]

# Ensure text index exists
collection.create_index([
    ("name", "text"),
    ("description", "text"),
    ("generatedSummary", "text")
], name="DocumentTextIndex", default_language='english')

# Mongo setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]  # type: ignore
collection = db["documents"]


@router.get("/search")
def search_documents(query: str = Query(..., min_length=1), request: Request = None):
    # User is authenticated through middleware, can access this endpoint
    user_id = request.state.user.get('username') if request and hasattr(request, 'state') else None
    
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
def get_documents_by_ids(ids: str = Query(..., description="Comma-separated document IDs"), request: Request = None):
    # User is authenticated through middleware, can access this endpoint
    user_id = request.state.user.get('username') if request and hasattr(request, 'state') else None
    
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
    description: str = Form(...),
    request: Request = None
):
    # User is authenticated through middleware, admin access required
    user_id = request.state.user.get('username') if request and hasattr(request, 'state') else None
    
    try:
        allowed_ext = [".doc", ".docx", ".pdf"]
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_ext:
            return JSONResponse(status_code=400, content={"error": f"File type {ext} not allowed"})

        # Check for duplicate
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
            "dateAdded": get_indian_time(),
            "uploaded_by": user_id  # Track who uploaded the document
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



@router.post("/reindex")
async def reindex_documents(data: dict = Body(...), request: Request = None):
    """Re-index already indexed documents"""
    # User is authenticated through middleware, admin access required
    user_id = request.state.user.get('username') if request and hasattr(request, 'state') else None
    
    doc_ids = data.get("doc_ids", [])
    if not doc_ids:
        return JSONResponse(status_code=400, content={"error": "Missing doc_ids"})

    # Import here to avoid circular imports
    from .indexing import index_document

    # Call the existing index_document function
    result = await index_document({"doc_ids": doc_ids})
    return result


@router.post("/add2")
async def add_document_v2(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(...),
    request: Request = None
):
    # User is authenticated through middleware, admin access required
    user_id = request.state.user.get('username') if request and hasattr(request, 'state') else None
    
    try:
        #  Check for duplicate
        if collection.find_one({"filename": file.filename}):
            return JSONResponse(status_code=400, content={"error": "File with this name already exists"})

        doc_id = str(uuid.uuid4())
        doc_dir = RESOURCE_DIR / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        file_path = doc_dir / file.filename  # type: ignore
        with open(file_path, "wb") as f:
            f.write(await file.read())

        collection.insert_one({
            "_id": doc_id,
            "name": name,
            "filename": file.filename,
            "description": description,
            "generatedSummary": "",
            "isIndexed": False,
            "dateAdded": get_indian_time(),
            "uploaded_by": user_id  # Track who uploaded the document
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
def delete_doc(data: dict = Body(...), request: Request = None):
    # User is authenticated through middleware, admin access required
    user_id = request.state.user.get('username') if request and hasattr(request, 'state') else None
    
    doc_id = data.get("doc_id")
    if not doc_id:
        return JSONResponse(status_code=400, content={"error": "Missing doc_id"})

    # Always remove from Mongo
    collection.delete_one({"_id": doc_id})

    # Delete folder if it exists
    doc_dir = RESOURCE_DIR / doc_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir, ignore_errors=True)

    return {"status": "deleted", "id": doc_id}


# @router.get("/list2")
# def list_documents():
#     docs = list(collection.find({}, {"_id": 1, "name": 1, "filename": 1, "isIndexed": 1, "description": 1, "generatedSummary": 1, "dateAdded": 1}))
#     for doc in docs:
#         doc["id"] = str(doc["_id"])
#         del doc["_id"]
#     return docs

@router.get("/download/{doc_id}/{filename}")
def download_document(doc_id: str, filename: str, request: Request = None):
    # User is authenticated through middleware, can access this endpoint
    user_id = request.state.user.get('username') if request and hasattr(request, 'state') else None
    
    doc_dir = RESOURCE_DIR / doc_id
    file_path = doc_dir / filename
    if not file_path.exists() or not file_path.is_file():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(str(file_path), filename=filename)


@router.get("/list")
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = Query(None, description="Optional search by name"),
    request: Request = None
):
    # User is authenticated through middleware, can access this endpoint
    user_id = request.state.user.get('username') if request and hasattr(request, 'state') else None
    
    skips = (page - 1) * page_size

    query_filter = {}
    if search:
        query_filter["name"] = {"$regex": search, "$options": "i"}

    total = collection.count_documents(query_filter)
    cursor = collection.find(
        query_filter,
        {"_id": 1, "name": 1, "filename": 1, "isIndexed": 1, "description": 1, "generatedSummary": 1, "dateAdded": 1}
    ).sort("dateAdded", -1).skip(skips).limit(page_size)

    docs = list(cursor)
    for idx, doc in enumerate(docs, start=skips + 1):
        doc["id"] = str(doc["_id"])
        doc["serial_no"] = idx
        del doc["_id"]

    return {"items": docs, "total": total, "page": page, "page_size": page_size}

