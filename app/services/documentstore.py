import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

class DocumentStore:
    def __init__(self, storage_dir="./data/documents"):
        """
        Initialize the document store for tracking uploaded documents.
        
        Args:
            storage_dir: Directory to store document metadata
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.metadata_file = os.path.join(storage_dir, "document_metadata.json")
        self.documents = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load document metadata from disk"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save document metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.documents, f, indent=2)
    
    def add_document(self, file_path: str, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the store and return its ID.
        
        Args:
            file_path: Path to the document file
            title: Optional document title
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        filename = os.path.basename(file_path)
        title = title or filename
        
        self.documents[doc_id] = {
            "id": doc_id,
            "title": title,
            "filename": filename,
            "file_path": file_path,
            "upload_date": datetime.now().isoformat(),
            "metadata": metadata or {},
            "num_chunks": 0,
            "status": "pending"
        }
        
        self._save_metadata()
        return doc_id
    
    def update_document_status(self, doc_id: str, status: str, num_chunks: int = None):
        """
        Update document processing status and chunk count.
        
        Args:
            doc_id: Document ID
            status: Processing status (pending, processing, completed, failed)
            num_chunks: Number of chunks created
        """
        if doc_id in self.documents:
            self.documents[doc_id]["status"] = status
            if num_chunks is not None:
                self.documents[doc_id]["num_chunks"] = num_chunks
            self._save_metadata()
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get document metadata by ID"""
        return self.documents.get(doc_id, {})
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with their metadata"""
        return list(self.documents.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from the store"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._save_metadata()
            return True
        return False