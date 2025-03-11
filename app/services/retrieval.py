from typing import List, Dict, Any
import json
import os

class RetrievalService:
    def __init__(self, embedding_service):
        """
        Initialize retrieval service with embedding service for vector search.
        """
        self.embedding_service = embedding_service
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a given query.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with metadata
        """
        retrieved_chunks = self.embedding_service.retrieve_similar_chunks(query, top_k)
        
        # Format the results
        results = []
        for chunk, idx in retrieved_chunks:
            metadata = self.embedding_service.metadata_store.get(idx, {})
            results.append({
                "content": chunk,
                "doc_id": metadata.get("doc_id", "unknown"),
                "score": float(idx),  # This should be replaced with actual similarity score
                "metadata": metadata
            })
            
        return results
    
    def save_index(self, file_path: str):
        """
        Save FAISS index to disk
        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save FAISS index
        faiss.write_index(self.embedding_service.index, f"{file_path}.index")
        
        # Save metadata
        with open(f"{file_path}.meta", 'w') as f:
            json.dump(self.embedding_service.metadata_store, f)
            
    def load_index(self, file_path: str):
        """
        Load FAISS index from disk
        """
        # Load FAISS index
        self.embedding_service.index = faiss.read_index(f"{file_path}.index")
        
        # Load metadata
        with open(f"{file_path}.meta", 'r') as f:
            self.embedding_service.metadata_store = json.load(f)