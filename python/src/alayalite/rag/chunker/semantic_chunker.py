# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module provides the SemanticChunker class, which splits text based on
semantic similarity using sentence embeddings.
"""

import os
import sys
from typing import List

import numpy as np
from rag.chunker.base import BaseChunker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)


class SemanticChunker(BaseChunker):
    """
    A dynamic text chunker based on semantic similarity.

    Args:
        model_name (str): Name of the semantic encoding model. Defaults to 'all-MiniLM-L6-v2'.
        threshold (float): Similarity threshold (0-1). Defaults to 0.8.
        window_size (int): Sliding window size in sentences. Defaults to 3.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.8, window_size: int = 3):
        """Initializes the Semantic Chunker."""
        # This class does not use the BaseChunker's __init__ as its logic is different.
        super().__init__(chunk_size=0, chunk_overlap=0)
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.window_size = window_size

    def chunking(self, text: str) -> List[str]:
        """
        Implements the semantic-aware chunking logic.

        Args:
            text (str): The input text.

        Returns:
            List[str]: A list of chunked text blocks.
        """
        # Step 1: Basic sentence splitting
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # Step 2: Calculate sentence embeddings
        embeddings = self._encode_sentences(sentences)

        # Step 3: Sliding window analysis
        chunks = []
        current_chunk_sentences = []
        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)

            # Start checking when enough sentences have accumulated
            if len(current_chunk_sentences) >= self.window_size:
                # Compare the similarity of the current window with the next one
                if self._should_split(embeddings, i):
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []

        # Add any remaining content as the last chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def _should_split(self, embeddings: np.ndarray, current_index: int) -> bool:
        """
        Determines whether to split by calculating semantic similarity using a sliding window.

        Calculation logic:
        previous_window = [sent_{k-n}, ..., sent_k]
        next_window = [sent_{k+1}, ..., sent_{k+n+1}]
        similarity = cos_sim(mean(prev_window), mean(next_window))
        """
        window_size = self.window_size

        # Get the index ranges for the current and next windows
        prev_start = max(0, current_index - window_size + 1)
        next_start = current_index + 1
        next_end = min(len(embeddings), current_index + window_size + 1)

        # If there's no next window to compare, don't split
        if next_start >= len(embeddings):
            return False

        # Calculate mean embeddings
        prev_emb = np.mean(embeddings[prev_start : current_index + 1], axis=0)
        next_emb = np.mean(embeddings[next_start:next_end], axis=0)

        # Calculate cosine similarity
        similarity = cosine_similarity([prev_emb], [next_emb])[0][0]
        return similarity < self.threshold

    def _encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Batch encode sentences into embedding vectors."""
        return self.model.encode(sentences, convert_to_numpy=True)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Basic sentence splitting (can be replaced with more complex logic if needed)."""
        # A simple split by period, can be improved with regex for more delimiters.
        return [s.strip() for s in text.split(".") if s.strip()]
