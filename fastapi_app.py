"""
PRODUCTION RAG API SERVER - FASTAPI
====================================
FastAPI backend for document upload, processing, and retrieval.
High-performance async API with automatic OpenAPI documentation.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import json
from pathlib import Path
from datetime import datetime
import logging

from rag_engine import RAGPipeline, RetrievalResult
from chatbot_engine import ProductionChatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Intelligence API",
    description="Production-grade Retrieval-Augmented Generation system with hybrid chunking and FAISS HNSW indexing",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = 'data/uploads'
CHUNKS_FOLDER = 'data/chunks'
INDEX_FOLDER = 'data/index'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Create folders
for folder in [UPLOAD_FOLDER, CHUNKS_FOLDER, INDEX_FOLDER]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Global state
rag_pipeline: Optional[RAGPipeline] = None
chatbot: Optional[ProductionChatbot] = None
document_registry: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query", min_length=1)
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)


class QueryResponse(BaseModel):
    success: bool
    query: str
    results_count: int
    results: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    chunks_count: int
    chunks_file: str
    message: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    documents_indexed: int


class StatsResponse(BaseModel):
    success: bool
    stats: Dict[str, Any]


class DocumentsResponse(BaseModel):
    success: bool
    documents: Dict[str, Any]


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1)
    conversation_id: str = Field("default", description="Conversation ID for context")


class ChatResponse(BaseModel):
    success: bool
    answer: str
    sources: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    conversation_id: str
    timestamp: str
    metadata: Dict[str, Any]


class ConversationHistoryResponse(BaseModel):
    success: bool
    conversation_id: str
    messages: List[Dict[str, Any]]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_rag_pipeline():
    """Initialize RAG pipeline (lazy loading)"""
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(
            embedding_model_name="all-MiniLM-L6-v2",
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        # Try to load existing index
        index_path = os.path.join(INDEX_FOLDER, 'faiss_index.bin')
        metadata_path = os.path.join(INDEX_FOLDER, 'metadata.pkl')
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                rag_pipeline.load_index(index_path, metadata_path)
                logger.info("Loaded existing index")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")
        
        logger.info("RAG pipeline ready")
    return rag_pipeline


def init_chatbot():
    """Initialize chatbot (lazy loading)"""
    global chatbot, rag_pipeline
    
    if chatbot is None:
        # Ensure RAG is initialized
        if rag_pipeline is None:
            init_rag_pipeline()
        
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        logger.info("Initializing chatbot with Groq Llama 8B Instant...")
        chatbot = ProductionChatbot(
            groq_api_key=groq_api_key,
            rag_pipeline=rag_pipeline,
            model_name="llama-3.1-8b-instant",
            temperature=0.7,
            max_context_chunks=5
        )
        logger.info("Chatbot ready")
    
    return chatbot


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "RAG Document Intelligence API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        documents_indexed=len(document_registry)
    )


@app.post("/api/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...)
):
    """
    Upload and process a document.
    
    - **file**: PDF, DOCX, or TXT file (max 16MB)
    
    Returns document metadata and chunk count.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024)}MB"
            )
        
        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
        unique_filename = f"{timestamp}_{safe_filename}"
        
        # Save file
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        with open(filepath, 'wb') as f:
            f.write(content)
        
        logger.info(f"Uploaded file: {unique_filename}")
        
        # Process document SYNCHRONOUSLY (so user gets immediate feedback)
        logger.info(f"Processing {safe_filename}...")
        rag = init_rag_pipeline()
        
        # Ingest and chunk document
        chunks = rag.ingest_document(filepath)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Build/update index
        rag.build_index(chunks)
        logger.info(f"Index updated")
        
        # Save chunks to JSON
        chunks_data = [
            {
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,
                'source': chunk.source,
                'paragraph_index': chunk.paragraph_index,
                'sub_chunk_index': chunk.sub_chunk_index,
                'token_count': chunk.token_count
            }
            for chunk in chunks
        ]
        
        chunks_filename = f"{timestamp}_{Path(safe_filename).stem}_chunks.json"
        chunks_path = os.path.join(CHUNKS_FOLDER, chunks_filename)
        
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save index
        index_path = os.path.join(INDEX_FOLDER, 'faiss_index.bin')
        metadata_path = os.path.join(INDEX_FOLDER, 'metadata.pkl')
        rag.save_index(index_path, metadata_path)
        
        # Update registry
        document_registry[unique_filename] = {
            'filename': safe_filename,
            'upload_time': timestamp,
            'chunks_count': len(chunks),
            'chunks_file': chunks_filename,
            'file_path': filepath
        }
        
        logger.info(f"Successfully processed {safe_filename}: {len(chunks)} chunks")
        
        # Return complete response with chunk count
        return UploadResponse(
            success=True,
            document_id=unique_filename,
            filename=safe_filename,
            chunks_count=len(chunks),
            chunks_file=chunks_filename,
            message=f"Document processed successfully with {len(chunks)} chunks"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse, tags=["Search"])
async def query_documents(request: QueryRequest):
    """
    Query the RAG system.
    
    - **query**: Your search query
    - **top_k**: Number of results to return (1-20)
    
    Returns relevant document chunks with scores.
    """
    try:
        # Initialize RAG pipeline
        rag = init_rag_pipeline()
        
        if rag.retriever is None:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please upload documents first."
            )
        
        # Query
        logger.info(f"Query: {request.query}")
        results = rag.query(request.query, top_k=request.top_k)
        
        # Format results
        results_data = [
            {
                'chunk_id': r.chunk_id,
                'text': r.text,
                'combined_score': float(r.combined_score),
                'embedding_score': float(r.embedding_score),
                'rerank_score': float(r.rerank_score),
                'metadata': r.metadata
            }
            for r in results
        ]
        
        return QueryResponse(
            success=True,
            query=request.query,
            results_count=len(results_data),
            results=results_data
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents", response_model=DocumentsResponse, tags=["Documents"])
async def list_documents():
    """List all processed documents"""
    return DocumentsResponse(
        success=True,
        documents=document_registry
    )


@app.get("/api/chunks/{filename}", tags=["Documents"])
async def get_chunks(filename: str):
    """Get chunks for a specific document"""
    try:
        chunks_path = os.path.join(CHUNKS_FOLDER, filename)
        
        if not os.path.exists(chunks_path):
            raise HTTPException(status_code=404, detail="Chunks file not found")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        return {
            'success': True,
            'chunks': chunks
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    """Get system statistics"""
    rag = init_rag_pipeline()
    
    total_chunks = 0
    if rag.faiss_index:
        total_chunks = rag.faiss_index.index.ntotal
    
    def get_folder_size(folder):
        total = 0
        try:
            for f in Path(folder).glob('**/*'):
                if f.is_file():
                    total += f.stat().st_size
        except:
            pass
        return total
    
    return StatsResponse(
        success=True,
        stats={
            'documents_uploaded': len(document_registry),
            'total_chunks': int(total_chunks),
            'index_dimension': rag.dimension,
            'upload_folder_size': get_folder_size(UPLOAD_FOLDER),
            'chunks_folder_size': get_folder_size(CHUNKS_FOLDER)
        }
    )


@app.delete("/api/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    try:
        if document_id not in document_registry:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_info = document_registry[document_id]
        
        # Delete files
        if os.path.exists(doc_info['file_path']):
            os.remove(doc_info['file_path'])
        
        chunks_path = os.path.join(CHUNKS_FOLDER, doc_info['chunks_file'])
        if os.path.exists(chunks_path):
            os.remove(chunks_path)
        
        # Remove from registry
        del document_registry[document_id]
        
        return {
            'success': True,
            'message': f'Document {document_id} deleted successfully'
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CHAT ENDPOINTS
# =============================================================================

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_documents(request: ChatRequest):
    """
    Chat with your documents using Groq Llama 8B Instant
    
    The chatbot:
    - Retrieves relevant context from indexed documents
    - Understands images and tables
    - Maintains conversation history
    - Provides sourced answers
    """
    try:
        # Initialize chatbot
        bot = init_chatbot()
        
        # Process chat
        response = bot.chat(
            message=request.message,
            conversation_id=request.conversation_id
        )
        
        return ChatResponse(
            success=True,
            answer=response["answer"],
            sources=response["sources"],
            images=response["images"],
            conversation_id=response["conversation_id"],
            timestamp=response["timestamp"],
            metadata=response["metadata"]
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history/{conversation_id}", response_model=ConversationHistoryResponse, tags=["Chat"])
async def get_conversation_history(conversation_id: str):
    """Get conversation history for a specific conversation"""
    try:
        bot = init_chatbot()
        history = bot.get_conversation_history(conversation_id)
        
        return ConversationHistoryResponse(
            success=True,
            conversation_id=conversation_id,
            messages=history
        )
    
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/history/{conversation_id}", tags=["Chat"])
async def clear_conversation_history(conversation_id: str):
    """Clear conversation history"""
    try:
        bot = init_chatbot()
        bot.clear_conversation(conversation_id)
        
        return {
            'success': True,
            'message': f'Conversation {conversation_id} cleared'
        }
    
    except Exception as e:
        logger.error(f"History clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/stats", tags=["Chat"])
async def get_chatbot_stats():
    """Get chatbot statistics"""
    try:
        bot = init_chatbot()
        stats = bot.get_stats()
        
        return {
            'success': True,
            'stats': stats
        }
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("RAG API SERVER (FastAPI)")
    print("=" * 80)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Chunks folder: {CHUNKS_FOLDER}")
    print(f"Index folder: {INDEX_FOLDER}")
    print("=" * 80)
    print("API Documentation: http://localhost:8000/api/docs")
    print("=" * 80)
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
