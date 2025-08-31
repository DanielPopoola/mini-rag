from typing import List, Dict, Any, Optional
import time
import logging

from .embeddings import EmbeddingModel
from .chunking import DocumentChunker
from .retrieval import RetrievalSystem
from ..services.vector_db import VectorDatabase
from ..services.llm import LocalLLM, GenerationResponse


logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_db: VectorDatabase,
        retrieval_system: RetrievalSystem,
        llm: LocalLLM,
        chunker: DocumentChunker
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.retrieval_system = retrieval_system
        self.llm = llm
        self.chunker = chunker

    def process_document(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document: chunk -> embed -> store
        """
        start_time = time.time()

        # Chunk the document
        logger.info("Chunking document...")
        chunks = self.chunker.chunk_text(text, metadata)

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode_batch(texts)

        # Store in vector database
        logger.info("Storing in vector database...")
        self.vector_db.upsert_chunks(chunks, embeddings)

        processing_time = time.time() - start_time

        return {
            "status": "success",
            "chunks_processed": len(chunks),
            "processing_time": processing_time,
            "source": metadata.get("source", "unknown")
        }

    def query(self, question: str, source_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve -> rerank -> generate
        """
        start_time = time.time()

        # Retrieve and rerank relevant chunks
        logger.info(f"Processing query: {question}")
        retrieved_chunks = self.retrieval_system.retrieve_and_rerank(
            query=question,
            source_filter=source_filter
        )

        retrieval_time = time.time() - start_time

        # Generate answer with LLM
        logger.info("Generating answer...")
        generation_start = time.time()

        response = self.llm.generate_answer(question, retrieved_chunks)

        generation_time = time.time() - generation_start
        total_time = time.time() - start_time

        # Return complete response with metrics
        return {
            "question": question,
            "answer": response.answer,
            "citations": response.citations,
            "confidence": response.confidence,
            "reasoning": response.reasoning,
            "retrieved_chunks": len(retrieved_chunks),
            "metrics": {
                "total_time": total_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "tokens_generated": response.token_count
            },
            "retrieval_stats": self.retrieval_system.get_retrieval_stats(retrieved_chunks)
        }
