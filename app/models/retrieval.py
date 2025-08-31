from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import CrossEncoder
import logging

from ..models.embeddings import EmbeddingModel
from ..services.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class RetrievalSystem:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_db: VectorDatabase,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_batch_size: int = 16,
        rerank_threshold: float = 0.0,
    ):
        """
        Retrieval system with bi-encoder vector search + cross-encoder reranking.
        
        Args:
            embedding_model: Model to encode queries/documents.
            vector_db: Vector database instance for similarity search.
            reranker_model: HuggingFace cross-encoder model for reranking.
            reranker_batch_size: Number of query-doc pairs to process at once in reranker.
        """
        self.embedding_model = embedding_model
        self.vector_db = vector_db

        # Cross-enocder for reranking
        self.reranker = CrossEncoder(reranker_model)
        self.reranker_batch_size = reranker_batch_size
        self.rerank_threshold = rerank_threshold

        # Retrieval parameters
        self.initial_k = 20
        self.final_k = 5

    def retrieve_and_rerank(
        self,
        query: str,
        source_filter: Optional[str] = None,
        final_k: Optional[int] = None,
        rerank_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Complete retrieval pipeline: embed -> search -> rerank.
        """
        final_k = final_k or self.final_k
        rerank_threshold = rerank_threshold or self.rerank_threshold

        # Convert query to embedding
        logger.info(f"Embedding query: {query[:50]}...")
        query_embedding = self.embedding_model.encode_batch([query], normalize=True)[0]

        # Vector search
        logger.info(f"Vector search for top-{self.initial_k} candidates")
        initial_results = self.vector_db.search_similar(
            query_embedding=query_embedding,
            top_k=self.initial_k,
            source_filter=source_filter,
        )

        if not initial_results:
            logger.warning("No results found from vector search")
            return []

        # Rerank
        logger.info(f"Reranking candidates to top-{final_k}")
        try:
            return self._rerank_results(query, initial_results, final_k, rerank_threshold)
        except Exception as e:
            logger.error(f"Reranker failed: {e}. Returning vector search results only.")
            return initial_results[:final_k]

    def _rerank_results(
        self,
        query: str,
        initial_results: List[Dict[str, Any]],
        final_k: int,
        rerank_threshold: float,
    ) -> List[Dict[str, Any]]:

        """
        Rerank results using cross-encoder for better relevance
        """
        query_doc_pairs = [[query, r["text"]] for r in initial_results]

        rerank_scores: List[float] = []
        # Process in batches to avoid OOM
        for i in range(0, len(query_doc_pairs), self.reranker_batch_size):
            batch = query_doc_pairs[i: i + self.reranker_batch_size]
            batch_scores = self.reranker.predict(batch)
            rerank_scores.extend(batch_scores)

        # Merge results with scores 
        for i, result in enumerate(initial_results):
            result["rerank_score"] = float(rerank_scores[i])
            result["original_score"] = float(result.get("score", 0.0))

        # Sort by rerank score
        reranked = sorted(initial_results, key=lambda x: x["rerank_score"], reverse=True)

        # Filter by threshold and return top-k
        filtered_results = [r for r in reranked if r["rerank_score"] >= rerank_threshold]
        
        logger.info(f"Reranking kept {len(filtered_results)} of {len(reranked)} results with threshold {rerank_threshold}")

        return filtered_results[:final_k]

    def get_retrieval_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute retrieval stats for diagnostics
        """
        if not results:
            return {"message": "No results to analyze"}

        rerank_scores = [r.get("rerank_score", 0.0) for r in results]
        original_scores = [r.get("original_score", 0.0) for r in results]

        return {
            "num_results": len(results),
            "avg_rerank_score": float(np.mean(rerank_scores)),
            "max_rerank_score": float(np.max(rerank_scores)),
            "min_rerank_score": float(np.min(rerank_scores)),
            "avg_original_score": float(np.mean(original_scores)),
            "score_improvement": float(np.mean(rerank_scores) - np.mean(original_scores)),
            "sources": list({r.get("metadata", {}).get("source", "unknown") for r in results}),
        }