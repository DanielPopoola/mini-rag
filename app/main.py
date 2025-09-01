from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import time
import traceback
from contextlib import asynccontextmanager


from .models.rag_pipeline import RAGPipeline
from .models.embeddings import EmbeddingModel
from .models.chunking import DocumentChunker
from .models.retrieval import RetrievalSystem
from .services.vector_db import VectorDatabase
from .services.llm import LocalLLM, OpenRouterLLM
from .config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - setup and teardown"""
    # Startup
    logger.info("🚀 Starting Mini RAG API...")

    try:
        # Initialize all components
        logger.info("Initializing embedding model...")
        embedding_model = EmbeddingModel()

        logger.info("Initializing vector database...")
        vector_db = VectorDatabase()
        vector_db.create_collection(dimension=embedding_model.dimension)

        logger.info("Initializing LLM...")
        llm = OpenRouterLLM(model_name="deepseek-r1:7b")

        logger.info("Initializing chunker and retrieval system...")
        chunker = DocumentChunker()
        retrieval_system = RetrievalSystem(
            embedding_model=embedding_model,
            vector_db=vector_db
        )

        logger.info("Creating RAG pipeline...")
        global pipeline
        pipeline = RAGPipeline(
            embedding_model=embedding_model,
            vector_db=vector_db,
            retrieval_system=retrieval_system,
            llm=llm,
            chunker=chunker
        )

        logger.info("✅ Mini RAG API ready!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG pipeline: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Mini RAG API...")


app = FastAPI(
    title="Mini RAG API",
    description="A lightweight RAG system",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],  # React, Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    text: str = Field(..., description="Document text content")
    source: str = Field(..., description="Source identifier (filename, URL, etc.)")
    title: Optional[str] = Field(None, description="Document title")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class QueryRequest(BaseModel):
    """Request model for querying"""
    question: str = Field(..., description="User's question", min_length=1)
    source_filter: Optional[str] = Field(None, description="Filter results by source")
    max_results: Optional[int] = Field(5, description="Maximum number of results", ge=1, le=20)

class Citation(BaseModel):
    """Citation information"""
    citation_id: int
    text: str
    source: str
    title: str
    rerank_score: float

class QueryResponse(BaseModel):
    """Response model for queries"""
    question: str
    answer: str
    citations: List[Citation]
    confidence: str
    reasoning: str
    retrieved_chunks: int
    metrics: Dict[str, Any]

class DocumentResponse(BaseModel):
    """Response model for document processing"""
    status: str
    message: str
    chunks_processed: int
    processing_time: float
    source: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    components: Dict[str, str]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Mini RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    components = {}

    try:
        # Check if pipeline is initialized
        if pipeline is None:
            components["pipeline"] = "not_initialized"
        else:
            components["pipeline"] = "healthy"

        try:
            info = pipeline.vector_db.get_collection_info()
            components["vector_db"] = f"healthy ({info['points_count']} documents)"
        except Exception:
            components["vector_db"] = "unhealthy"

        try:
            test_response = pipeline.llm._call_openrouter("Test", max_tokens = 5)
            components["llm"] = "healthy"
        except Exception:
            components["llm"] = "unhealthy"

    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

    return HealthResponse(
        status="healthy" if all(status != "unhealthy" for status in components.values()) else "degraded",
        timestamp=time.time(),
        components=components
    )

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(request: DocumentUploadRequest):
    """Process and store a document"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        logger.info(f"Processing document: {request.source}")

        metadata  = {
            "source": request.source,
            "title": request.title or request.source,
            **request.metadata
        }

        result = pipeline.process_document(request.text, metadata)

        return DocumentResponse(
            status="success",
            message=f"Successfully processed document '{request.source}'",
            chunks_processed=result["chunks_processed"],
            processing_time=result["processing_time"],
            source=result["source"]
        )

    except Exception as e:
        logger.error(f"Document processing error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/documents/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    """Upload and a process a file"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        content = await file.read()

        if file.filename.endswith(".txt") or file.filename.endswith(".md"):
            text = content.decode("utf-8")
        elif file.filename.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(content)
            return "".join(page.extract_text() for page in reader.pages)
        elif file.filename.endswith(".docx"):
            import docx
            doc = docx.Document(content)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

        # Process using the text upload endpoint logic
        metadata = {
            "source": file.filename,
            "title": title or file.filename,
            "file_type": file.content_type
        }

        result = pipeline.process_document(text, metadata)
        
        return DocumentResponse(
            status="success",
            message=f"Successfully processed file '{file.filename}'",
            chunks_processed=result["chunks_processed"],
            processing_time=result["processing_time"],
            source=result["source"]
        )
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection"""

    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        logger.info(f"Processing query: {request.question}")

        # Execute query
        result = pipeline.query(question=request.question, source_filter=request.source_filter)

        # Convert to response model
        citations = [
            Citation(
                citation_id=cite["citation_id"],
                text=cite["text"],
                source=cite["source"],
                title=cite["title"],
                rerank_score=cite["rerank_score"]
            )
            for cite in result["citations"]
        ]

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            citations=citations,
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            retrieved_chunks=result["retrieved_chunks"],
            metrics=result["metrics"]
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/documents/stats")
async def get_document_stats():
    """Get statistics about stored documents"""

    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        info = pipeline.vector_db.get_collection_info()
        return {
            "total_documents": info["points_count"],
            "total_vectors": info["vectors_count"],
            "indexed_vectors": info["indexed_vectors_count"],
            "collection_name": pipeline.vector_db.collection_name
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.delete("/documents/{source}")
async def delete_documents(source: str):
    """Delete all chunks from a specific document source"""
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        pipeline.vector_db.delete_by_source(source)
        return {"status": "success", "message": f"Deleted document: {source}"}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)