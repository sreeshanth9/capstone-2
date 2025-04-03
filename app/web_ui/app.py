import streamlit as st
import requests
import json
import os
import time
from typing import Dict, List, Any
import pandas as pd

# Configuration
API_URL = "http://localhost:8000/api"  # Adjust if your API is hosted elsewhere

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Helper Functions ----
def upload_file(file):
    """Upload a file to the API"""
    files = {"file": (file.name, file.getvalue(), f"application/{file.type}")}
    response = requests.post(f"{API_URL}/upload", files=files)
    return response.json()

def process_document(doc_id):
    """Trigger document processing"""
    response = requests.post(f"{API_URL}/process/{doc_id}")
    return response.json()

def get_documents():
    """Get list of all documents"""
    response = requests.get(f"{API_URL}/documents")
    return response.json().get("documents", [])

def query_rag(query, top_k=5, include_sources=True):
    """Send a query to the RAG pipeline"""
    payload = {
        "query": query,
        "top_k": top_k,
        "include_sources": include_sources
    }
    response = requests.post(f"{API_URL}/query", json=payload)
    return response.json()

# ---- Session State Initialization ----
if "documents" not in st.session_state:
    st.session_state.documents = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# ---- UI Layout ----
st.title("📚 RAG Document Assistant")

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
    
    if uploaded_file is not None:
        if st.button("Upload Document"):
            with st.spinner("Uploading document..."):
                try:
                    result = upload_file(uploaded_file)
                    st.success(f"Document uploaded: {uploaded_file.name}")
                    st.session_state.current_doc_id = result.get("document_id")
                    
                    # Immediately start processing
                    st.session_state.processing = True
                    process_btn_placeholder = st.empty()
                    process_btn_placeholder.info("Processing document... Please wait.")
                    
                    try:
                        process_result = process_document(st.session_state.current_doc_id)
                        st.session_state.processing = False
                        process_btn_placeholder.success("✅ Document processed successfully!")
                    except Exception as e:
                        st.session_state.processing = False
                        process_btn_placeholder.error(f"Error processing document: {str(e)}")
                        
                    # Refresh documents list
                    st.session_state.documents = get_documents()
                except Exception as e:
                    st.error(f"Error uploading document: {str(e)}")
    
    # Document selector
    st.subheader("Your Documents")
    if st.button("Refresh Documents"):
        with st.spinner("Loading documents..."):
            st.session_state.documents = get_documents()
    
    if st.session_state.documents:
        doc_options = {doc["title"]: doc["id"] for doc in st.session_state.documents}
        selected_doc = st.selectbox(
            "Select a document",
            options=list(doc_options.keys()),
            index=0
        )
        if selected_doc:
            st.session_state.current_doc_id = doc_options[selected_doc]
            
            # Show document info
            for doc in st.session_state.documents:
                if doc["id"] == st.session_state.current_doc_id:
                    st.write(f"Status: {doc.get('status', 'Unknown')}")
                    st.write(f"Chunks: {doc.get('num_chunks', 0)}")
                    st.write(f"Upload date: {doc.get('upload_date', 'Unknown')}")
    else:
        st.info("No documents available. Upload one to get started!")

# Main content area
main_col1, main_col2 = st.columns([2, 3])

with main_col1:
    st.header("Ask Questions")
    
    query = st.text_area("Enter your question", height=100)
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=100, value=5)
    with col2:
        include_sources = st.checkbox("Include sources", value=True)
    
    if st.button("Submit Question"):
        if not query:
            st.warning("Please enter a question.")
        elif st.session_state.processing:
            st.warning("Please wait for document processing to complete.")
        elif st.session_state.current_doc_id is None:
            st.warning("Please upload or select a document first.")
        else:
            with st.spinner("Generating response..."):
                try:
                    response = query_rag(query, top_k, include_sources)
                    
                    # Add to query history
                    st.session_state.query_history.append({
                        "query": query,
                        "response": response,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    st.error(f"Error querying RAG: {str(e)}")

with main_col2:
    st.header("Responses")
    
    if st.session_state.query_history:
        # Show most recent query first
        for i, item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"Q: {item['query']}", expanded=(i == 0)):
                st.markdown("### Answer")
                st.write(item["response"].get("response", "No response"))
                
                if include_sources and "sources" in item["response"]:
                    st.markdown("### Sources")
                    sources = item["response"].get("sources", [])
                    if sources:
                        for src in sources:
                            doc_id = src.get("doc_id", "Unknown")
                            doc_title = "Unknown"
                            
                            # Find document title
                            for doc in st.session_state.documents:
                                if doc["id"] == doc_id:
                                    doc_title = doc["title"]
                                    break
                                    
                            st.info(f"Document: {doc_title}")
                            
                            # If there's metadata to display
                            metadata = src.get("metadata", {})
                            if metadata:
                                st.json(metadata)
                st.caption(f"Asked at {item['timestamp']}")
    else:
        st.info("No questions asked yet. Start by asking a question!")

# Footer
st.markdown("---")
st.caption("RAG Document Assistant | Powered by FastAPI & Streamlit")