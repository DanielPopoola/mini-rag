from __future__ import annotations
from typing import List, Dict, Any
from transformers import AutoTokenizer
import nltk
import logging


logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Sentence-aware, token-accurate chunker with overlap for RAG.


    Design goals:
    - Use the *same tokenizer* as your embedding model for accurate token counts.
    - Keep chunks <= chunk_size tokens (content tokens, no special tokens).
    - Maintain semantic coherence by splitting on sentence boundaries when possible.
    - Add token-level overlap between consecutive chunks to avoid boundary loss.
    - Safely handle very long sentences (> chunk_size) by token-slicing with stride.


    Typical usage:
    chunker = DocumentChunker(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=512,
    overlap_size=50,
    )
    chunks = chunker.chunk_text(text, metadata={"doc_id": "my_doc"})


    Each returned chunk is a dict: {"text": str, "metadata": {...}}.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        overlap_size: int = 50,
        sentence_separator: str = " ",        
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.chunk_size = int(chunk_size)
        self.overlap_size = int(overlap_size)
        self.sentence_separator = sentence_separator
        if not ( 0 < self.overlap_size < self.chunk_size):
            raise ValueError("overlap_size must be > 0 and < chunk_size")
        try:
            nltk.data.find("tokenizers/punkt")
        except Exception:
            nltk.download("punkt")

    def _token_count(self, text: str) -> int:
        """Returns the number of tokens in a string"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        return nltk.sent_tokenize(text)

    def _slice_long_sentence(self, sentence: str) -> List[str]:
        """Split a very long sentence (> chunk_size tokens) into token windows with overlap."""
        token_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        n = len(token_ids)
        if n <= self.chunk_size:
            return [sentence]
        stride = max(self.chunk_size - self.overlap_size, 1)
        parts: List[str] = []
        for start in range(0, n, stride):
            end = min(start + self.chunk_size, n)
            piece_ids = token_ids[start:end]
            piece_text = self.tokenizer.decode(
                piece_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            if piece_text:
                parts.append(piece_text)
            if end >= n:
                break
        return parts

    def _ensure_sentences_units(self, text: str) -> List[str]:
        """Split into sentences, and further split any that exceed chunk_size tokens."""
        base = self._split_into_sentences(text)
        logger.info(f"Split text into {len(base)} sentences.")
        result: List[str] = []
        for s in base:
            if self._token_count(s) > self.chunk_size:
                result.extend(self._slice_long_sentence(s))
            else:
                result.append(s)
        return result

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert raw text into overlapping, sentence-aware token-constrained chunks.

        Args:
        text: Raw document text.
        metadata: Base metadata dict (e.g., {"doc_id": "...", "source": "..."}).


        Returns:
        List of {"text": str, "metadata": Dict[str, Any]} chunks.
        """
        sentences = self._ensure_sentences_units(text)
        tok_lens = [self._token_count(s) for s in sentences]

        chunks: List[Dict[str, Any]] = []
        i = 0
        chunk_idx = 0

        while i < len(sentences):
            total = 0
            j = i
            # Greedily add sentences while staying within chunk_size
            while j < len(sentences) and total + tok_lens[j] <= self.chunk_size:
                total += tok_lens[j]
                j += 1

            # Safety: if nothing fits (shouldn't happen due to _slice_long_sentence), force-slice
            if j == i:
                long_subparts = self._slice_long_sentence(sentences[i])
                for part in long_subparts:
                    if not part.strip():
                        continue
                    part_tokens = self._token_count(part)
                    meta = {
                        **metadata,
                        "chunk_index": chunk_idx,
                        "num_tokens": part_tokens,
                        "start_sentence": i,
                        "end_sentence": i,
                        "overlap": self.overlap_size,
                        }
                    chunks.append({"text": part, "metadata": meta})
                    chunk_idx += 1
                i += 1
                continue

            # Build the chunk from sentences [i, j)
            chunk_text = self.sentence_separator.join(sentences[i:j]).strip()
            chunk_tokens = self._token_count(chunk_text)
            meta = {
                **metadata,
                "chunk_index": chunk_idx,
                "num_tokens": chunk_tokens,
                "start_sentence": i,
                "end_sentence": j - 1,
                "overlap": self.overlap_size,
            }
            chunks.append({"text": chunk_text, "metadata": meta})
            chunk_idx += 1

            if  j >= len(sentences):
                break

            # Advance window with token-overlap. Keep ~overlap_size tokens from the tail of this chunk.
            tokens_to_step = max(chunk_tokens - self.overlap_size, 1)
            walked = 0
            k = i
            while k < j and walked + tok_lens[k] <= tokens_to_step:
                walked += tok_lens[k]
                k += 1
            i = k

        # Stamp stable chunk_ids using doc_id if available
        doc_id = str(metadata.get("doc_id", metadata.get("source_id", "doc")))
        for ch in chunks:
            ch["metadata"]["chunk_id"] = f"{doc_id}::{ch['metadata']['chunk_index']}"

        return chunks


__all__ = ["DocumentChunker"]