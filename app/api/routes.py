import os
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from pymongo import MongoClient
from services.embeddings import EmbeddingService
from services.chunking import Chunking
from database.mongodb import get_database

client = MongoClient("mongodb://localhost:27017/")
db = client["file_uploads"]
collection = db["uploads"]

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()
chunking_service = Chunking()
embedding_service = EmbeddingService()
db = get_database()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1].lower()
    allowed_types = {"pdf", "txt", "docx"}

    if file_extension not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_type_folder = os.path.join(UPLOAD_DIR, file_extension)
    os.makedirs(file_type_folder, exist_ok=True)

    file_path = os.path.join(file_type_folder, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    file_metadata = {
        "filename": file.filename,
        "file_type": file_extension,
        "upload_timestamp": datetime.utcnow().isoformat(),
        "file_path": file_path.replace("\\", "/"),  # Ensure cross-platform compatibility
    }

    collection.insert_one(file_metadata)

    return {"message": "File uploaded successfully", "metadata": file_metadata}

@router.post("/process/{file_id}")
def process_file(file_id: str):
    file_metadata = db["uploads"].find_one({"_id": file_id})
    if not file_metadata:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = file_metadata["file_path"]
    
    chunks = chunking_service.process(file_path)

    embedding_service.generate_and_store_embeddings(chunks)

    return {"message": "File processing completed successfully", "chunks": len(chunks)}