import os
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query, Body
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from .schemas import QueryRequest, QueryResponse

# Import your services
from app.services.chunking import ChunkingService
from app.services.embeddings import EmbeddingService
from app.services.retrieval import RetrievalService
from app.services.llm import LLMService
from app.database.mongodb import get_database
from app.services.mongodb_document_store import MongoDBDocumentStore
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize services
chunking_service = ChunkingService()
embedding_service = EmbeddingService()
retrieval_service = RetrievalService(embedding_service)
llm_service = LLMService()

# Get database connection and initialize document store
db = get_database()
document_store = MongoDBDocumentStore(db)

# Create RAG pipeline
rag_pipeline = RAGPipeline(
    chunking_service=chunking_service,
    embedding_service=embedding_service,
    retrieval_service=retrieval_service,
    llm_service=llm_service,
    document_store=document_store
)

# Try to load existing index
try:
    retrieval_service.load_index("./data/index/vector_index")
    print(f"Loaded existing vector index")
except Exception as e:
    print(f"No existing index found or error loading: {e}")


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    file_extension = file.filename.split(".")[-1].lower()
    allowed_types = {"pdf", "txt", "docx"}

    if file_extension not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_type_folder = os.path.join(UPLOAD_DIR, file_extension)
    os.makedirs(file_type_folder, exist_ok=True)

    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(file_type_folder, unique_filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Create file metadata
    file_metadata = {
        "filename": file.filename,
        "file_type": file_extension,
        "upload_timestamp": datetime.utcnow(),
        "file_path": file_path.replace("\\", "/"),  # Ensure cross-platform compatibility
        "status": "pending"
    }

    # Insert into MongoDB directly using the document store
    doc_id = document_store.add_document(
        file_path=file_path, 
        title=file.filename,
        metadata=file_metadata
    )
    
    # Process in background if background_tasks is provided
    if background_tasks:
        background_tasks.add_task(process_document, file_path, doc_id)
        return {
            "message": "File uploaded successfully. Processing in background.", 
            "document_id": doc_id
        }
    
    return {
        "message": "File uploaded successfully",
        "document_id": doc_id,
        "metadata": file_metadata
    }

@router.post("/process/{doc_id}")
def process_document_endpoint(doc_id: str):
    document = document_store.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = document.get("file_path")
    
    try:
        rag_pipeline.process_document(file_path, doc_id=doc_id)
        return {"message": "Document processing completed successfully", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

def process_document(file_path: str, doc_id: str):
    """Background task to process a document"""
    try:
        document_store.update_document_status(doc_id, "processing")
        rag_pipeline.process_document(file_path, doc_id=doc_id)
        document_store.update_document_status(doc_id, "completed")
    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")
        document_store.update_document_status(doc_id, f"failed: {str(e)}")

@router.get("/documents")
def list_documents():
    """List all documents"""
    documents = document_store.list_documents()
    return {"documents": documents}

@router.get("/documents/{doc_id}")
def get_document(doc_id: str):
    """Get document details"""
    document = document_store.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document"""
    success = document_store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "message": "Document deleted"}

@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Process a query through the RAG pipeline"""
    try:
        result = rag_pipeline.query(
            request.query, 
            top_k=request.top_k, 
            include_sources=request.include_sources
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# import os
# import uuid
# from datetime import datetime
# from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query, Body
# from typing import Optional, Dict, Any, List
# from pymongo import MongoClient
# from pydantic import BaseModel
# from .schemas import QueryRequest, QueryResponse

# # Import your services
# from app.services.chunking import ChunkingService
# from app.services.embeddings import EmbeddingService
# from app.services.retrieval import RetrievalService
# from app.services.llm import LLMService
# from app.services.document_store import DocumentStore
# from app.services.rag_pipeline import RAGPipeline
# from app.database.mongodb import get_database

# # MongoDB setup
# client = MongoClient("mongodb://localhost:27017/")
# db = client["file_uploads"]
# collection = db["uploads"]

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# router = APIRouter()
# d
# # Initialize RAG pipeline
# chunking_service = ChunkingService()
# embedding_service = EmbeddingService()
# retrieval_service = RetrievalService(embedding_service)
# document_store = DocumentStore()
# llm_service = LLMService()

# # Create RAG pipeline
# rag_pipeline = RAGPipeline(
#     chunking_service=chunking_service,
#     embedding_service=embedding_service,
#     retrieval_service=retrieval_service,
#     llm_service=llm_service,
#     document_store=document_store
# )

# # Try to load existing index
# try:
#     retrieval_service.load_index("./data/index/vector_index")
#     print(f"Loaded existing vector index")
# except Exception as e:
#     print(f"No existing index found or error loading: {e}")


# @router.post("/upload")
# async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
#     file_extension = file.filename.split(".")[-1].lower()
#     allowed_types = {"pdf", "txt", "docx"}

#     if file_extension not in allowed_types:
#         raise HTTPException(status_code=400, detail="Unsupported file type.")

#     file_type_folder = os.path.join(UPLOAD_DIR, file_extension)
#     os.makedirs(file_type_folder, exist_ok=True)

#     # Generate unique filename
#     unique_filename = f"{uuid.uuid4()}_{file.filename}"
#     file_path = os.path.join(file_type_folder, unique_filename)

#     # Save file
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # Create file metadata
#     file_metadata = {
#         "filename": file.filename,
#         "file_type": file_extension,
#         "upload_timestamp": datetime.utcnow().isoformat(),
#         "file_path": file_path.replace("\\", "/"),  # Ensure cross-platform compatibility
#         "status": "pending"
#     }

#     # Insert into MongoDB
#     result = collection.insert_one(file_metadata)
#     file_id = str(result.inserted_id)
    
#     # Also add to document store
#     doc_id = document_store.add_document(
#         file_path=file_path, 
#         title=file.filename,
#         metadata={"mongo_id": file_id}
#     )
    
#     # Process in background if background_tasks is provided
#     if background_tasks:
#         background_tasks.add_task(process_document, file_path, doc_id)
#         return {
#             "message": "File uploaded successfully. Processing in background.", 
#             "document_id": doc_id,
#             "mongo_id": file_id
#         }
    
#     return {
#         "message": "File uploaded successfully",
#         "document_id": doc_id,
#         "mongo_id": file_id,
#         "metadata": file_metadata
#     }

# @router.post("/process/{doc_id}")
# def process_document_endpoint(doc_id: str):
#     document = document_store.get_document(doc_id)
#     if not document:
#         raise HTTPException(status_code=404, detail="Document not found")

#     file_path = document.get("file_path")
    
#     try:
#         rag_pipeline.process_document(file_path, doc_id=doc_id)
#         return {"message": "Document processing completed successfully", "document_id": doc_id}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# def process_document(file_path: str, doc_id: str):
#     """Background task to process a document"""
#     try:
#         rag_pipeline.process_document(file_path, doc_id=doc_id)
#     except Exception as e:
#         print(f"Error processing document {doc_id}: {e}")
#         document_store.update_document_status(doc_id, f"failed: {str(e)}")

# @router.get("/documents")
# def list_documents():
#     """List all documents"""
#     documents = document_store.list_documents()
#     return {"documents": documents}

# @router.get("/documents/{doc_id}")
# def get_document(doc_id: str):
#     """Get document details"""
#     document = document_store.get_document(doc_id)
#     if not document:
#         raise HTTPException(status_code=404, detail="Document not found")
#     return document

# @router.delete("/documents/{doc_id}")
# def delete_document(doc_id: str):
#     """Delete a document"""
#     success = document_store.delete_document(doc_id)
#     if not success:
#         raise HTTPException(status_code=404, detail="Document not found")
#     return {"status": "success", "message": "Document deleted"}

# @router.post("/query", response_model=QueryResponse)
# def query(request: QueryRequest):
#     """Process a query through the RAG pipeline"""
#     try:
#         result = rag_pipeline.query(
#             request.query, 
#             top_k=request.top_k, 
#             include_sources=request.include_sources
#         )
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

# @router.get("/health")
# def health_check():
#     """Health check endpoint"""
#     return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
