# from typing import Dict, List, Any, Optional
# import os

# class RAGPipeline:
#     def __init__(self, chunking_service, embedding_service, retrieval_service, llm_service, document_store):
#         self.chunking_service = chunking_service
#         self.embedding_service = embedding_service
#         self.retrieval_service = retrieval_service
#         self.llm_service = llm_service
#         self.document_store = document_store
        
#     def process_document(self, file_path: str, doc_id: str):
#         """
#         Process a document:
#         1. Extract text and chunk it
#         2. Generate embeddings and add to vector index
#         3. Store chunks in MongoDB
#         """
#         # Extract text and chunk it
#         chunks = self.chunking_service.process(file_path)
        
#         # For each chunk, create a document with appropriate metadata
#         chunk_documents = []
        
#         for i, chunk in enumerate(chunks):
#             # import pdb; pdb.set_trace()
#             chunk_text = chunk["text"]
#             chunk_document = {
#                 "chunk_id": f"{doc_id}_{i}",
#                 "text": chunk_text,
#                 "document_id": doc_id,
#                 "index": i,
#                 "metadata": {
#                     "source": file_path,
#                     "page": chunk.get("metadata", {}).get("page", None),
#                     "position": i
#                 }
#             }
            
#             # Generate embedding for the chunk
#             embedding = self.embedding_service.get_embedding(chunk_text)
#             chunk_document["embedding"] = embedding
            
#             chunk_documents.append(chunk_document)
        
#         # Add chunks to MongoDB
#         self.document_store.add_chunks(doc_id, chunk_documents)
        
#         # Add vectors to retrieval index
#         self.retrieval_service.add_texts(
#             texts=[chunk["text"] for chunk in chunk_documents],
#             embeddings=[chunk["embedding"] for chunk in chunk_documents],
#             metadatas=[{
#                 "document_id": chunk["document_id"],
#                 "chunk_id": chunk["chunk_id"],
#                 "source": chunk["metadata"]["source"]
#             } for chunk in chunk_documents]
#         )
        
#         # Save the updated index
#         self.retrieval_service.save_index("./data/index/vector_index")
        
#         return len(chunk_documents)
        
#     def query(self, query_text: str, top_k: int = 3, include_sources: bool = True):
#         """
#         Process a query through the RAG pipeline:
#         1. Retrieve relevant chunks from the vector index
#         2. Generate response using LLM with retrieved context
#         """
#         # Get query embedding
#         query_embedding = self.embedding_service.get_embedding(query_text)
        
#         # Retrieve relevant chunks
#         search_results = self.retrieval_service.search_by_vector(
#             query_embedding, top_k=top_k
#         )
        
#         # Format context from search results
#         context_texts = [result["text"] for result in search_results]
#         context = "\n\n".join(context_texts)
        
#         # Generate response
#         response = self.llm_service.generate_response(query_text, context)
        
#         # Prepare sources if requested
#         sources = None
#         if include_sources:
#             sources = []
#             for result in search_results:
#                 doc_id = result["metadata"]["document_id"]
#                 document = self.document_store.get_document(doc_id)
                
#                 sources.append({
#                     "document_id": doc_id,
#                     "title": document.get("title", "Unknown"),
#                     "text": result["text"][:200] + "...",  # Preview text
#                     "relevance": float(result["score"]) if "score" in result else None
#                 })
        
#         return {
#             "query": query_text,
#             "response": response,
#             "sources": sources
#         }

# class RAGPipeline:
#     def __init__(self, chunking_service, embedding_service, retrieval_service, llm_service, document_store):
#         """
#         Initialize the complete RAG pipeline.
        
#         Args:
#             chunking_service: Service for document chunking
#             embedding_service: Service for embedding generation
#             retrieval_service: Service for chunk retrieval
#             llm_service: Service for response generation
#             document_store: Service for document management
#         """
#         self.chunking_service = chunking_service
#         self.embedding_service = embedding_service
#         self.retrieval_service = retrieval_service
#         self.llm_service = llm_service
#         self.document_store = document_store
        
#     def process_document(self, file_path: str, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, doc_id=None) -> str:
#         """
#         Process a document through the RAG pipeline.
        
#         Args:
#             file_path: Path to the document file
#             title: Optional document title
#             metadata: Additional metadata
            
#         Returns:
#             Document ID
#         """
#         # Add document to store
#         doc_id = self.document_store.add_document(file_path, title, metadata)
        
#         try:
#             # Update status
#             self.document_store.update_document_status(doc_id, "processing")
            
#             # Extract text
#             text = self.chunking_service.extract_text(file_path)
#             cleaned_text = self.chunking_service.clean_text(text)
            
#             # Create chunks
#             sentences = self.chunking_service.split_into_sentences(cleaned_text)
#             chunks = self.chunking_service.recursive_chunking(sentences)
            
#             # Store embeddings
#             self.embedding_service.store_embeddings(chunks, doc_id)
            
#             # Update document status
#             self.document_store.update_document_status(doc_id, "completed", len(chunks))
            
#             return doc_id
            
#         except Exception as e:
#             # Handle errors
#             self.document_store.update_document_status(doc_id, f"failed: {str(e)}")
#             raise
    
#     def query(self, query: str, top_k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
#         """
#         Process a query through the RAG pipeline.
        
#         Args:
#             query: User query
#             top_k: Number of chunks to retrieve
#             include_sources: Whether to include source information
            
#         Returns:
#             Response with optional source information
#         """
#         # Retrieve relevant chunks
#         context_chunks = self.retrieval_service.retrieve(query, top_k)
        
#         # Generate response
#         if include_sources:
#             return self.llm_service.generate_response_with_sources(query, context_chunks)
#         else:
#             response = self.llm_service.generate_response(query, context_chunks)
#             return {"response": response}


from typing import Dict, List, Any, Optional
import os

class RAGPipeline:
    def __init__(self, chunking_service, embedding_service, retrieval_service, llm_service, document_store):
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service
        self.document_store = document_store
        
    def process_document(self, file_path: str, doc_id: str):
        """
        Process a document:
        1. Extract text and chunk it
        2. Generate embeddings and add to vector index
        3. Store chunks in MongoDB
        """
        # Extract text and chunk it
        chunks = self.chunking_service.process(file_path)
        chunk_texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings and store in FAISS
        embeddings = self.embedding_service.store_embeddings(chunk_texts, doc_id)

        print(f"Number of embeddings stored: {len(embeddings)}")
        print(f"Index total vectors: {self.embedding_service.index.ntotal}")

        # Create chunk documents with embeddings
        chunk_documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_document = {
                "chunk_id": f"{doc_id}_{i}",
                "text": chunk["text"],
                "document_id": doc_id,
                "index": i,
                "metadata": {
                    "source": file_path,
                    "page": chunk.get("metadata", {}).get("page", None),
                    "position": i
                },
                "embedding": embedding.tolist()  # Ensure it's stored as a list
            }
            chunk_documents.append(chunk_document)

        # Store chunks in MongoDB
        self.document_store.add_chunks(doc_id, chunk_documents)

        # Add vectors to retrieval index
        self.retrieval_service.add_texts(
            texts=[chunk["text"] for chunk in chunk_documents],
            embeddings=[chunk["embedding"] for chunk in chunk_documents],
            metadatas=[{
                "document_id": chunk["document_id"],
                "chunk_id": chunk["chunk_id"],
                "source": chunk["metadata"]["source"]
            } for chunk in chunk_documents]
        )

        # Save the updated index
        self.retrieval_service.save_index("./data/index/vector_index")

        return len(chunk_documents)
        
    def query(self, query_text: str, top_k: int = 3, include_sources: bool = True):
        """
        Process a query through the RAG pipeline:
        1. Retrieve relevant chunks from the vector index
        2. Generate response using LLM with retrieved context
        """
        # Get query embedding
        query_embedding = self.embedding_service.generate_embeddings([query_text])[0]

        # Retrieve relevant chunks
        search_results = self.retrieval_service.search_by_vector(
            query_embedding, top_k=top_k
        )

        # Format context from search results
        context_texts = [result["text"] for result in search_results]
        context = "\n\n".join(context_texts)

        print("Search results:", search_results)

        # Generate response
        response = self.llm_service.generate_response(query_text, search_results)

        # Prepare sources if requested
        sources = None
        if include_sources:
            sources = []
            for result in search_results:
                doc_id = result["metadata"]["document_id"]
                document = self.document_store.get_document(doc_id)

                sources.append({
                    "document_id": doc_id,
                    "title": document.get("title", "Unknown"),
                    "text": result["text"][:200] + "...",  # Preview text
                    "relevance": float(result["score"]) if "score" in result else None
                })

        return {
            "query": query_text,
            "response": response,
            "sources": sources
        }
