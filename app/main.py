import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your API router
from app.api.routes import router

# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="Retrieval Augmented Generation API for document processing and querying",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix="/api")

# Ensure necessary directories exist
os.makedirs("./data/documents", exist_ok=True)
os.makedirs("./data/uploads", exist_ok=True)
os.makedirs("./data/index", exist_ok=True)

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the RAG API",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json"
    }

if __name__ == "__main__":
    # Check for OpenAI API key
    # if not os.environ.get("OPENAI_API_KEY"):
    #     print("Warning: OPENAI_API_KEY environment variable not set")
    #     print("Set it with: export OPENAI_API_KEY='your-key-here'")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # For development
    )