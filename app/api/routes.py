import os
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from pymongo import MongoClient
from services.chunking import Chunking

client = MongoClient("mongodb://localhost:27017/")
db = client["file_uploads"]
collection = db["uploads"]

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

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
