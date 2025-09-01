from typing import List, Dict, Any, Optional, Union, Tuple
import json
import logging
import requests
from dataclasses import dataclass
import openai
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

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
        model_name: str = "gemma3:1b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        Initialize connection to local Ollama instance
        
        Args:
            model_name: Ollama model (llama3.1:8b, mistral:7b, etc.)
            base_url: Ollama API endpoint
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.timeout = timeout

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
            6. Your response MUST be a valid JSON object. DO NOT include any other text or conversational filler.

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
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
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
            end = raw_response.rfind('}') + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_str = raw_response[start:end]
            parsed = json.loads(json_str)

            # Build citations list
            citations = []
            cited_sources = parsed.get("cited_sources", [])

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

class OpenRouterLLM:
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b:free", 
        api_key: str = None,
        timeout: int = 120
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.timeout = timeout

    def generate_answer(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        max_tokens: int = 1000
    ) -> GenerationResponse:
        if not retrieved_chunks:
            return GenerationResponse(
                answer="I don't have enough information to answer this question.",
                citations=[],
                confidence="no_answer",
                reasoning="No relevant chunks were retrieved",
                token_count=0
            )

        system_prompt, user_prompt = self._build_rag_prompt(query, retrieved_chunks)

        combined_prompt = (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"USER:\n{user_prompt}"
        )

        try:
            raw_response, token_count = self._call_openrouter(combined_prompt, max_tokens)

            return self._parse_response(raw_response, retrieved_chunks, token_count)
        except Exception as e:
            logger.error(f"Error in generate_answer: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")

    def _build_rag_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> tuple[str, str]:
        system_prompt = ( """
        You are a helpful AI assistant that answers questions based on provided context.
        Follow these rules:
        1. Answer the question using ONLY the information provided in the context.
        2. Use inline citations [1], [2], etc. that correspond to the numbered sources.
        3. If the context doesn't contain enough information, say so clearly.
        4. Provide a confidence level: high, medium, low, or no_answer.
        5. Be concise but complete.
        Respond in the exact JSON format specified by the user.
        """ )

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"]["source"]
            title = chunk["metadata"].get("title", "Unknown")
            context_parts.append(f"[{i}] {chunk['text']}\nSource: {title} ({source})")
        context = "\n\n".join(context_parts)

        user_prompt = f"""
            CONTEXT:
            {context}

            JSON Response Format:
            {{
                "answer": "Your answer here with inline citations [1], [2]...",
                "confidence": "high/medium/low/no_answer",
                "reasoning": "Brief explanation of your confidence level",
                "cited_sources": [1, 2, ...]
            }}

            QUESTION: {query}
        """
        return system_prompt, user_prompt

    def _call_openrouter(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: int) -> Tuple[str, int]:
        """Make request to OpenRouter API"""
        try:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1
            )

            response_text = completion.choices[0].message.content
            token_count = getattr(completion.usage, "total_tokens", 0)

            return response_text, token_count

        except Exception as e:
            logger.error(f"Error calling OpenRouter: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")

    def _parse_response(self, raw_response: str, chunks: List[Dict[str, Any]], token_count: int) -> GenerationResponse:
        try:
            parsed = json.loads(raw_response)

            citations = []
            cited_sources = parsed.get("cited_sources", [])

            for source_num in cited_sources:
                if 1 <= source_num <= len(chunks):
                    chunk = chunks[source_num - 1]
                    citations.append({
                        "citation_id": source_num,
                        "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                        "source": chunk["metadata"]["source"],
                        "title": chunk["metadata"].get("title", "Unknown"),
                        "rerank_score": chunk.get("rerank_score", 0.0)
                    })

            return GenerationResponse(
                answer=parsed.get("answer", "No answer provided"),
                citations=citations,
                confidence=parsed.get("confidence", "low"),
                reasoning=parsed.get("reasoning", "No reasoning provided"),
                token_count=token_count
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {raw_response[:500]}...")
            return GenerationResponse(
                answer=raw_response,
                citations=[],
                confidence="low",
                reasoning="Failed to parse structured response from API.",
                token_count=token_count
            )
