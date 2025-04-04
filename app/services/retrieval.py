# from typing import List, Dict, Any
# import json
# import os
# import faiss

# class RetrievalService:
#     def __init__(self, embedding_service):
#         """
#         Initialize retrieval service with embedding service for vector search.
#         """
#         self.embedding_service = embedding_service
        
#     def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """
#         Retrieve relevant chunks for a given query.
        
#         Args:
#             query: User query string
#             top_k: Number of chunks to retrieve
            
#         Returns:
#             List of retrieved chunks with metadata
#         """
#         retrieved_chunks = self.embedding_service.retrieve_similar_chunks(query, top_k)
        
#         # Format the results
#         results = []
#         for chunk, idx in retrieved_chunks:
#             metadata = self.embedding_service.metadata_store.get(idx, {})
#             results.append({
#                 "content": chunk,
#                 "doc_id": metadata.get("doc_id", "unknown"),
#                 "score": float(idx),  # This should be replaced with actual similarity score
#                 "metadata": metadata
#             })
            
#         return results
    
#     def save_index(self, file_path: str):
#         """
#         Save FAISS index to disk
#         """
#         directory = os.path.dirname(file_path)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
            
#         # Save FAISS index
#         faiss.write_index(self.embedding_service.index, f"{file_path}.index")
        
#         # Save metadata
#         with open(f"{file_path}.meta", 'w') as f:
#             json.dump(self.embedding_service.metadata_store, f)
            
#     def load_index(self, file_path: str):
#         """
#         Load FAISS index from disk
#         """
#         # Load FAISS index
#         self.embedding_service.index = faiss.read_index(f"{file_path}.index")
        
#         # Load metadata
#         with open(f"{file_path}.meta", 'r') as f:
#             self.embedding_service.metadata_store = json.load(f)



# from typing import List, Dict, Any
# import json
# import os
# import faiss
# import numpy as np

# class RetrievalService:
#     def __init__(self, embedding_service):
#         """
#         Initialize retrieval service with embedding service for vector search.
#         """
#         self.embedding_service = embedding_service

#     def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """
#         Retrieve relevant chunks for a given query.

#         Args:
#             query: User query string
#             top_k: Number of chunks to retrieve

#         Returns:
#             List of retrieved chunks with metadata
#         """
#         retrieved_chunks = self.embedding_service.retrieve_similar_chunks(query, top_k)

#         # Format the results
#         results = []
#         for chunk, idx in retrieved_chunks:
#             metadata = self.embedding_service.metadata_store.get(idx, {})
#             results.append({
#                 "content": chunk,
#                 "doc_id": metadata.get("doc_id", "unknown"),
#                 "score": float(idx),  # This should be replaced with actual similarity score
#                 "metadata": metadata
#             })

#         return results

#     def search_by_vector(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
#         """
#         Search for relevant chunks using FAISS index with a query embedding.

#         Args:
#             query_embedding: The vector representation of the query
#             top_k: Number of top results to return

#         Returns:
#             List of retrieved chunks with metadata
#         """
#         if self.embedding_service.index.ntotal == 0:
#             return []

#         # Search the FAISS index
#         _, indices = self.embedding_service.index.search(np.array([query_embedding], dtype=np.float32), top_k)

#         # Format the results
#         results = []
#         for idx in indices[0]:
#             if idx in self.embedding_service.metadata_store:
#                 metadata = self.embedding_service.metadata_store[idx]
#                 results.append({
#                     "text": metadata["chunk"],
#                     "metadata": metadata,
#                     "score": float(idx)  # Placeholder score (FAISS returns distances, not similarity)
#                 })

#         return results

#     def save_index(self, file_path: str):
#         """
#         Save FAISS index to disk
#         """
#         directory = os.path.dirname(file_path)
#         if not os.path.exists(directory):
#             os.makedirs(directory)

#         # Save FAISS index
#         faiss.write_index(self.embedding_service.index, f"{file_path}.index")

#         # Save metadata
#         with open(f"{file_path}.meta", 'w') as f:
#             json.dump(self.embedding_service.metadata_store, f)

#     def load_index(self, file_path: str):
#         """
#         Load FAISS index from disk
#         """
#         # Load FAISS index
#         self.embedding_service.index = faiss.read_index(f"{file_path}.index")

#         # Load metadata
#         with open(f"{file_path}.meta", 'r') as f:
#             self.embedding_service.metadata_store = json.load(f)


from typing import List, Dict, Any
import json
import os
import faiss
import numpy as np

class RetrievalService:
    def __init__(self, embedding_service):
        """
        Initialize retrieval service with embedding service for vector search.
        """
        self.embedding_service = embedding_service

    def add_texts(self, texts: List[str], embeddings: List[np.ndarray], metadatas: List[Dict[str, Any]]):
        """
        Add text embeddings to the FAISS index along with metadata.

        Args:
            texts: List of text chunks to be indexed.
            embeddings: Corresponding embeddings for the text chunks.
            metadatas: Metadata information for each text chunk.
        """
        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("Texts, embeddings, and metadatas must have the same length.")

        for i, (text, emb, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            idx = len(self.embedding_service.metadata_store)  # Get new index for FAISS
            self.embedding_service.index.add(np.array([emb], dtype=np.float32))
            self.embedding_service.metadata_store[idx] = {
                "chunk": text,
                "doc_id": metadata["document_id"],
                "chunk_id": metadata["chunk_id"],
                "source": metadata["source"]
            }

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
                "score": float(idx),  # Placeholder score
                "metadata": metadata
            })

        return results

    def search_by_vector(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using FAISS index with a query embedding.

        Args:
            query_embedding: The vector representation of the query
            top_k: Number of top results to return

        Returns:
            List of retrieved chunks with metadata
        """
        if self.embedding_service.index.ntotal == 0:
            return []

        # Search the FAISS index
        _, indices = self.embedding_service.index.search(np.array([query_embedding], dtype=np.float32), top_k)

        # Format the results
        results = []
        for idx in indices[0]:
            if idx in self.embedding_service.metadata_store:
                metadata = self.embedding_service.metadata_store[idx]
                results.append({
                    "text": metadata["chunk"],
                    "metadata": metadata,
                    "score": float(idx)  # Placeholder score
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
