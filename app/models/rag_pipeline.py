from typing import List, Dict, Any, Optional
import time
import logging

from .embeddings import EmbeddingModel
from .chunking import DocumentChunker
from .retrieval import RetrievalSystem
from ..services.vector_db import VectorDatabase
from ..services.llm import LocalLLM, GenerationResponse, OpenRouterLLM


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

    def query(self, question: str, source_filter: Optional[str] = None, rerank_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve -> rerank -> generate
        """
        start_time = time.time()

        # Retrieve and rerank relevant chunks
        logger.info(f"Processing query: {question}")
        retrieved_chunks = self.retrieval_system.retrieve_and_rerank(
            query=question,
            source_filter=source_filter,
            rerank_threshold=rerank_threshold
        )

        retrieval_time = time.time() - start_time

        # Generate answer with LLM
        logger.info("Generating answer...")
        generation_start = time.time()

        response = self.llm.generate_answer(question, retrieved_chunks, 1000)

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



def test_integration():
    try:
        # Initialize components exactly like Streamlit does
        print("1. Creating embedding model...")
        embedding_model = EmbeddingModel()
        
        print("2. Creating vector DB...")
        vector_db = VectorDatabase()
        vector_db.create_collection(dimension=384)
        
        print("3. Creating LLM...")
        llm = LocalLLM()
        
        print("4. Creating chunker...")
        chunker = DocumentChunker()  # Use default constructor
        
        print("5. Creating retrieval system...")
        retrieval_system = RetrievalSystem(
            embedding_model=embedding_model,
            vector_db=vector_db
        )
        
        print("6. Creating RAG pipeline...")
        rag_pipeline = RAGPipeline(
            embedding_model=embedding_model,
            vector_db=vector_db,
            retrieval_system=retrieval_system,
            llm=llm,
            chunker=chunker
        )
        
        print("7. Testing document processing...")
        with open("/home/daniel/mini-rag/seplat.txt", "r") as f:
            test_text = f.read()
        test_metadata = {"source": "seplat.txt", "title": "Seplat"}
        
        result = rag_pipeline.process_document(test_text, test_metadata)
        print(f"✅ Success! Result: {result}")

        print("8. Testing query...")
        question = "What is the proposal for?"
        query_result = rag_pipeline.query(question)
        print(f"✅ Success! Query Result: {query_result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()
