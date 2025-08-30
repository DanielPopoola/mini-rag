from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models
import uuid
import logging
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Qdrant client
        For local: url="localhost", port=6333  
        For cloud: url="https://your-cluster-url", api_key="your-key"
        """
        url = url or os.getenv("QDRANT_URL", "localhost")
        api_key = api_key or os.getenv("QDRANT_API_KEY")

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = "documents"

    def create_collections(self, dimension: int = 384, force_recreate: bool = False):
        """Create collection optimized for our embedding model"""

        # Check if collection exists
        collections = self.client.get_collections()
        collection_exists = any(col.name == self.collection_name for col in collections.collections)

        if collection_exists and force_recreate:
            logger.info(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            collection_exists = False

        if not collection_exists:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collections(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                )
            )

            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=models.KeywordIndexParams(
                    type="keyword"
                )
            )

            logger.info(f"Collection created successfully")
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Insert chunks with their embeddings
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: Numpy array of shape (len(chunks), dimension)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunk, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "chunk_id": chunk["chunk_id"], 
                    "token_count": chunk["token_count"],
                    "source": chunk["metadata"].get("source", "unknown"),
                    "title": chunk["metadata"].get("title", ""),
                    "chunk_position": chunk["metadata"]["chunk_position"],
                }
            )
            points.append(point)

        # Batch upsert for efficiency
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Upserted {len(points)} chunks to collection")

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 20,
                    source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query vector (1D numpy array)
            top_k: Number of results to return
            source_filter: Optional filter by document source
            
        Returns:
            List of dictionaries with chunk data and similarity scores
        """
        # Prepare search filter if needed
        search_filter = None
        if source_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter)
                    )
                ]
            )

        # Peform similarity search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )

        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload["text"],
                "metadata": {
                    "source": result.payload["source"],
                    "title": result.payload["title"], 
                    "chunk_position": result.payload["chunk_position"],
                    "token_count": result.payload["token_count"]
                }
            })

        logger.info(f"Found {len(results)} similar chunks")
        return results

    def delete_by_source(self, source: str):
        """Delete all chunks from a specific source (for updates)"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted all chunks from source: {source}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get info about the collection"""
        info = self.client.get_collection(self.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
        }

__all__ = ["VectorDatabase"]