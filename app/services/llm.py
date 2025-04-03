# import os
# from typing import List, Dict, Any
# from transformers import AutoModelForCausalLM, AutoTokenizer

# class LLMService:
#     def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", device="cuda"):
#         """
#         Initialize LLM service for generating responses using a local open source model.
        
#         Args:
#             model_name: HuggingFace model name/path to use
#             device: Device to run model on ("cuda", "mps", or "cpu")
#         """
#         self.model_name = model_name
#         self.device = device
        
#         # Load model and tokenizer
#         print(f"Loading model {model_name}...")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name, 
#             torch_dtype="auto",
#             device_map=device
#         )
#         print("Model loaded successfully")
        
#     def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
#         """
#         Generate response using local LLM with retrieved context.
        
#         Args:
#             query: User query
#             context_chunks: Retrieved context chunks
            
#         Returns:
#             Generated response
#         """
#         # Prepare context from retrieved chunks
#         context = "\n\n".join([chunk["content"] for chunk in context_chunks])
        
#         # Create prompt (adjust based on the model's expected format)
#         prompt = f"""
#         Answer the following question based on the provided context information. 
#         If you cannot answer the question based on the context, just say that you don't know.
        
#         Context:
#         {context}
        
#         Question: {query}
        
#         Answer:
#         """
        
#         # Generate response
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
#         # Generate with sampling
#         output = self.model.generate(
#             **inputs,
#             max_new_tokens=500,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True
#         )
        
#         # Decode and clean response
#         response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
#         # Extract just the answer part (this may need adjustment based on model output)
#         response = response.split("Answer:")[-1].strip()
        
#         return response

#     def generate_response_with_sources(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Generate response using LLM with retrieved context and include source information.
        
#         Args:
#             query: User query
#             context_chunks: Retrieved context chunks
            
#         Returns:
#             Dict containing generated response and sources
#         """
#         response = self.generate_response(query, context_chunks)
        
#         # Extract source information
#         sources = []
#         for chunk in context_chunks:
#             if chunk["doc_id"] not in [s["doc_id"] for s in sources]:
#                 sources.append({
#                     "doc_id": chunk["doc_id"],
#                     "metadata": chunk.get("metadata", {})
#                 })
        
#         return {
#             "response": response,
#             "sources": sources
#         }
















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


import os
import requests
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Retrieve Hugging Face API credentials
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")

class LLMService:
    def __init__(self):
        """
        Initialize LLM service using Hugging Face Inference API.
        """
        if not HF_TOKEN or not HF_ENDPOINT_URL:
            raise ValueError("Missing Hugging Face API credentials. Check your .env file.")

        self.headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

    # def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
    #     """
    #     Generate response using Hugging Face API with retrieved context.
        
    #     Args:
    #         query: User query
    #         context_chunks: Retrieved context chunks
            
    #     Returns:
    #         Generated response string
    #     """
    #     # Prepare context from retrieved chunks
    #     context = "\n\n".join([chunk["content"] for chunk in context_chunks])

    #     # Create input prompt
    #     prompt = f"""
    #     Answer the following question based on the provided context information. 
    #     If you cannot answer the question based on the context, just say that you don't know.

    #     Context:
    #     {context}

    #     Question: {query}

    #     Answer:
    #     """

    #     payload = {
    #         "inputs": prompt,
    #         "parameters": {
    #             "max_new_tokens": 500,
    #             "temperature": 0.7,
    #             "top_p": 0.9,
    #             "do_sample": True
    #         }
    #     }

    #     try:
    #         response = requests.post(HF_ENDPOINT_URL, headers=self.headers, data=json.dumps(payload))
    #         response.raise_for_status()
    #         result = response.json()

    #         # Extract generated text
    #         generated_text = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
    #         return generated_text.strip()
        
    #     except requests.exceptions.RequestException as e:
    #         print(f"Error calling Hugging Face API: {e}")
    #         return "Error: Unable to generate response."
        
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate response using Hugging Face API with retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated response string
        """
        # Prepare context from retrieved chunks with source identifiers
        formatted_contexts = []
        for i, chunk in enumerate(context_chunks):
            chunk_text = chunk["content"]
            doc_id = chunk.get("doc_id", f"Document {i+1}")
            formatted_contexts.append(f"[Source: {doc_id}]\n{chunk_text}")
        
        context = "\n\n---\n\n".join(formatted_contexts)
        
        # Create improved input prompt
        prompt = f"""
        You are a helpful assistant that provides accurate information based on the given context.
        
        CONTEXT INFORMATION:
        --------------------
        {context}
        --------------------
        
        USER QUERY: {query}
        
        Instructions:
        1. Answer the query based ONLY on the provided context information.
        2. If the context doesn't contain enough information to fully answer the query, explain what's missing.
        3. If the context contains contradictory information, point this out and explain the different perspectives.
        4. Use a direct, concise writing style while being comprehensive.
        5. If appropriate, structure your answer with bullet points or numbered lists for clarity.
        6. Do not make up information that isn't supported by the context.
        7. Do not refer to the sources explicitly in your answer (e.g., don't say "According to Source 1...").
        
        ANSWER:
        """
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        try:
            response = requests.post(HF_ENDPOINT_URL, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            # Extract generated text
            generated_text = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
            # Clean up the output - get only the part after "ANSWER:"
            if "ANSWER:" in generated_text:
                generated_text = generated_text.split("ANSWER:")[1].strip()
            return generated_text
        
        except requests.exceptions.RequestException as e:
            print(f"Error calling Hugging Face API: {e}")
            return "Error: Unable to generate response."


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
