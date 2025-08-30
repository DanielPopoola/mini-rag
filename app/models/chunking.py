from __future__ import annotations
from typing import List, Dict, Any
import re
from transformers import AutoTokenizer


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

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Lightweight sentence splitter with basic abbreviation handling.
        Falls back to paragraph-level if punctuation is scarce.
        """
        # Normalize excessive spaces; keep newlines for paragraph clues
        text = re.sub(r"[\t]+", " ", text)
        paragraphs = re.split(r"\n{2,}", text.strip())
        sentences: List[str] = []

        abbr = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S|U\.K|No)\."
        pattern = re.compile(
            rf"""
            ( # capture a sentence
            (?: # start with something that's not just a terminator
            (?!{abbr}) # don't split right after common abbreviations
            [^\n.!?…] # not an immediate terminator
            | [^.!?…] # general fallback
            )
            .*? # minimally match content
            (?:[.!?…]+(?=\s)|$) # end with terminator(s) or end of string
            )
            """,
            re.VERBOSE | re.UNICODE | re.DOTALL,
        )

        for para in paragraphs:
            p = para.strip()
            if not p:
                continue
            found = pattern.findall(p)
            if found:
                sentences.extend(s.strip() for s in found if s.strip())
            else:
                sentences.append(p)
        return sentences

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
