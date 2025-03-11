import os
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMService:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", device="cuda"):
        """
        Initialize LLM service for generating responses using a local open source model.
        
        Args:
            model_name: HuggingFace model name/path to use
            device: Device to run model on ("cuda", "mps", or "cpu")
        """
        self.model_name = model_name
        self.device = device
        
        # Load model and tokenizer
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto",
            device_map=device
        )
        print("Model loaded successfully")
        
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate response using local LLM with retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated response
        """
        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk["content"] for chunk in context_chunks])
        
        # Create prompt (adjust based on the model's expected format)
        prompt = f"""
        Answer the following question based on the provided context information. 
        If you cannot answer the question based on the context, just say that you don't know.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with sampling
        output = self.model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Decode and clean response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the answer part (this may need adjustment based on model output)
        response = response.split("Answer:")[-1].strip()
        
        return response

    def generate_response_with_sources(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate response using LLM with retrieved context and include source information.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Dict containing generated response and sources
        """
        response = self.generate_response(query, context_chunks)
        
        # Extract source information
        sources = []
        for chunk in context_chunks:
            if chunk["doc_id"] not in [s["doc_id"] for s in sources]:
                sources.append({
                    "doc_id": chunk["doc_id"],
                    "metadata": chunk.get("metadata", {})
                })
        
        return {
            "response": response,
            "sources": sources
        }
















# if api key available for gpt model we can go with this.
'''
import os
from typing import List, Dict, Any
import openai

class LLMService:
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """
        Initialize LLM service for generating responses.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.model = model
        
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate response using LLM with retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated response
        """
        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk["content"] for chunk in context_chunks])
        
        # Create prompt
        prompt = f"""
        Answer the following question based on the provided context information. 
        If you cannot answer the question based on the context, just say that you don't know.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

    def generate_response_with_sources(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate response using LLM with retrieved context and include source information.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Dict containing generated response and sources
        """
        response = self.generate_response(query, context_chunks)
        
        # Extract source information
        sources = []
        for chunk in context_chunks:
            if chunk["doc_id"] not in [s["doc_id"] for s in sources]:
                sources.append({
                    "doc_id": chunk["doc_id"],
                    "metadata": chunk.get("metadata", {})
                })
        
        return {
            "response": response,
            "sources": sources
        }
'''