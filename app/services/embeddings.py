# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from typing import List, Tuple

# class EmbeddingService:
#     def __init__(self, model_name="BAAI/bge-base-en", index_type="IVF-PQ", dimension=768, nlist=100):
#         """
#         Initialize embedding model and FAISS index.
#         """
#         self.model = SentenceTransformer(model_name)
#         self.dimension = dimension
#         self.index = self._initialize_faiss_index(index_type, dimension, nlist)
#         self.metadata_store = {}  # Store metadata alongside embeddings
    
#     def _initialize_faiss_index(self, index_type: str, dimension: int, nlist: int):
#         """
#         Create FAISS index based on the chosen type.
#         """
#         if index_type == "HNSW":
#             index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW for high-accuracy search
#         else:
#             quantizer = faiss.IndexFlatL2(dimension)  # Quantizer for IVF-PQ
#             index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 16, 8)  # IVF-PQ for scalability
#             index.train(np.random.random((1000, dimension)).astype("float32"))  # Dummy training
#         return index
    
#     def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
#         """
#         Generate embeddings for a list of text chunks.
#         """
#         return self.model.encode(texts, convert_to_numpy=True)
    
#     def store_embeddings(self, chunks: List[str], doc_id: str):
#         """
#         Generate and store embeddings in FAISS with metadata.
#         """
#         embeddings = self.generate_embeddings(chunks)
        
#         for i, emb in enumerate(embeddings):
#             idx = len(self.metadata_store)
#             self.index.add(np.array([emb], dtype=np.float32))
#             self.metadata_store[idx] = {"doc_id": doc_id, "chunk": chunks[i]}
    
#     def retrieve_similar_chunks(self, query: str, top_k=5) -> List[Tuple[str, float]]:
#         """
#         Retrieve top-k relevant chunks based on similarity search and rerank them.
#         """
#         query_embedding = self.generate_embeddings([query])[0]
        
#         _, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
#         retrieved_chunks = [(self.metadata_store[idx]["chunk"], idx) for idx in indices[0] if idx in self.metadata_store]
        
#         # Rerank using cosine similarity (optional hybrid reranking could be added here)
#         retrieved_chunks = sorted(retrieved_chunks, key=lambda x: np.dot(query_embedding, self.generate_embeddings([x[0]])[0]), reverse=True)
        
#         return retrieved_chunks[:top_k]
    
# # Usage example:
# # embedding_service = Embeddings()
# # embedding_service.store_embeddings(["Chunk 1 text", "Chunk 2 text"], doc_id="123")
# # results = embedding_service.retrieve_similar_chunks("Find relevant information")
# # print(results)



import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class EmbeddingService:
    def __init__(self, model_name="BAAI/bge-base-en", index_type="IVF-PQ", dimension=768, nlist=100):
        """
        Initialize embedding model and FAISS index.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension
        self.index = self._initialize_faiss_index(index_type, dimension, nlist)
        self.metadata_store = {}  # Store metadata alongside embeddings
    
    def _initialize_faiss_index(self, index_type: str, dimension: int, nlist: int):
        """
        Create FAISS index based on the chosen type.
        """
        if index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW for high-accuracy search
        else:
            quantizer = faiss.IndexFlatL2(dimension)  # Quantizer for IVF-PQ
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 16, 8)  # IVF-PQ for scalability
            index.train(np.random.random((1000, dimension)).astype("float32"))  # Dummy training
        return index
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text chunks.
        """
        return self.model.encode(texts, convert_to_numpy=True)
    
    def store_embeddings(self, chunks: List[str], doc_id: str) -> List[np.ndarray]:
        """
        Generate and store embeddings in FAISS with metadata.
        """
        embeddings = self.generate_embeddings(chunks)
        
        for i, emb in enumerate(embeddings):
            idx = len(self.metadata_store)
            self.index.add(np.array([emb], dtype=np.float32))
            self.metadata_store[idx] = {"doc_id": doc_id, "chunk": chunks[i]}
        
        return embeddings  # Now returns embeddings to be used in RAGPipeline
    
    def retrieve_similar_chunks(self, query: str, top_k=5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k relevant chunks based on similarity search and rerank them.
        """
        query_embedding = self.generate_embeddings([query])[0]
        
        _, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        retrieved_chunks = [(self.metadata_store[idx]["chunk"], idx) for idx in indices[0] if idx in self.metadata_store]
        
        # Rerank using cosine similarity (optional hybrid reranking could be added here)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda x: np.dot(query_embedding, self.generate_embeddings([x[0]])[0]), reverse=True)
        
        return retrieved_chunks[:top_k]
