from typing import Dict, List, Any, Optional
import os

class RAGPipeline:
    def __init__(self, chunking_service, embedding_service, retrieval_service, llm_service, document_store):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            chunking_service: Service for document chunking
            embedding_service: Service for embedding generation
            retrieval_service: Service for chunk retrieval
            llm_service: Service for response generation
            document_store: Service for document management
        """
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service
        self.document_store = document_store
        
    def process_document(self, file_path: str, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a document through the RAG pipeline.
        
        Args:
            file_path: Path to the document file
            title: Optional document title
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        # Add document to store
        doc_id = self.document_store.add_document(file_path, title, metadata)
        
        try:
            # Update status
            self.document_store.update_document_status(doc_id, "processing")
            
            # Extract text
            text = self.chunking_service.extract_text(file_path)
            cleaned_text = self.chunking_service.clean_text(text)
            
            # Create chunks
            sentences = self.chunking_service.split_into_sentences(cleaned_text)
            chunks = self.chunking_service.recursive_chunking(sentences)
            
            # Store embeddings
            self.embedding_service.store_embeddings(chunks, doc_id)
            
            # Update document status
            self.document_store.update_document_status(doc_id, "completed", len(chunks))
            
            return doc_id
            
        except Exception as e:
            # Handle errors
            self.document_store.update_document_status(doc_id, f"failed: {str(e)}")
            raise
    
    def query(self, query: str, top_k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            include_sources: Whether to include source information
            
        Returns:
            Response with optional source information
        """
        # Retrieve relevant chunks
        context_chunks = self.retrieval_service.retrieve(query, top_k)
        
        # Generate response
        if include_sources:
            return self.llm_service.generate_response_with_sources(query, context_chunks)
        else:
            response = self.llm_service.generate_response(query, context_chunks)
            return {"response": response}