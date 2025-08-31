from typing import List, Dict, Any, Optional
import json
import logging
import requests
from dataclasses import dataclass
import openai


logger = logging.getLogger(__name__)


@dataclass
class GenerationResponse:
    """Structured response from LLM"""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: str # "high", "medium", "low", "no_answer"
    reasoning: str
    token_count: int


class LocalLLM:
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1"
    ):
        """
        Initialize connection to local Ollama instance
        
        Args:
            model_name: Ollama model (llama3.1:8b, mistral:7b, etc.)
            base_url: Ollama API endpoint
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

        # Test connection and ensure model is available
        self._ensure_model_available()

    def _ensure_model_available(self):
        """Check if model is available and pull if needed"""
        try:
            # List available models
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]

            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available: {available_models}")
                logger.info(f"Pulling model {self.model_name}...")

                # Pull the model
                pull_response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name}
                )
                pull_response.raise_for_status()

            logger.info(f"Model {self.model_name} is ready")
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            raise ConnectionError("Ollama is not running or not accessible")

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_tokens: 1000
    ) -> GenerationResponse:
        """
        Generate answer with citations from retrieved chunks
        """
        if not retrieved_chunks:
            return GenerationResponse(
                answer="I don't have enough information to answer this question.",
                citations=[],
                confidence="no_answer",
                reasoning="No relevant chunks were retrieved",
                token_count=0
            )
        
        # Build the prompt with retrieved context
        prompt = self._build_rag_prompt(query, retrieved_chunks)

        # Generation response
        raw_response = self._call_ollama(prompt, max_tokens)

        # Parse structured response
        parsed_response = self._parse_response(raw_response, retrieved_chunks)

        return parsed_response

    def _build_rag_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Build a structured prompt that encourages proper citation
        """
        # Build context section with numbered references
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"]["source"]
            title = chunk["metadata"].get("title", "Unknown")

            context_parts.append(f"[{i}] {chunk['text']}\nSource: {title} ({source})")

        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful AI assistant that answers questions based on provided context.
        Follow these rules:

            CONTEXT:
            {context}

            INSTRUCTIONS:
            1. Answer the question using ONLY the information provided in the context above
            2. Use inline citations [1], [2], etc. that correspond to the numbered sources
            3. If the context doesn't contain enough information, say so clearly
            4. Provide a confidence level: high, medium, low, or no_answer
            5. Be concise but complete

            Respond in this exact JSON format:
            {{
                "answer": "Your answer here with inline citations [1], [2]...",
                "confidence": "high/medium/low/no_answer",
                "reasoning": "Brief explanation of your confidence level",
                "cited_sources": [1, 2, ...]
            }}

            QUESTION: {query}

            RESPONSE:"""
        
        return prompt

    def _call_ollama(self, prompt: str, max_tokens: int) -> str:
        """Make request to Ollama API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")

    def _parse_response(self, raw_response: str, chunks: List[Dict[str, Any]]) -> GenerationResponse:
        """
        Parse the structured JSON response from LLM
        """
        try:
            # Try to extract JSON from response
            start = raw_response.find('{')
            end = raw_response.find('}') + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_str = raw_response[start:end]
            parsed = json.loads(json_str)

            # Build citations list
            citations = []
            cited_sources = parsed.get("cited_source", [])

            for source_num in cited_sources:
                if 1 <= source_num <= len(chunks):
                    chunk = chunks[source_num - 1]  # Convert to 0-based index
                    citations.append({
                        "citation_id": source_num,
                        "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                        "source": chunk["metadata"]["source"],
                        "title": chunk["metadata"].get("title", "Unkown"),
                        "rerank_score": chunk.get("rerank_score", 0.0)
                    })

            return GenerationResponse(
                answer=parsed.get("answer", "No answer provided"),
                citations=citations,
                confidence=parsed.get("confidence", "low"),
                reasoning=parsed.get("reasoning", "No reasoning provided"),
                token_count=len(raw_response.split())
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {raw_response[:500]}...")

            # Fallback response
            return GenerationResponse(
                answer=raw_response,
                citations=[],
                confidence="low",
                reasoning="Failed to parse structured response",
                token_count=len(raw_response.split())
            )