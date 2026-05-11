# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides the JinaEmbedder class for creating embeddings using Jina AI models.
"""

from typing import List, Tuple

from sentence_transformers import SentenceTransformer

from .base import BaseEmbedding


class JinaEmbedder(BaseEmbedding):
    """An embedding class that uses Jina AI's sentence-transformer models."""

    def __init__(self, path: str = "jinaai/jina-embeddings-v2-base-en") -> None:
        """
        Initializes the JinaEmbedder.

        Args:
            path (str): The model path or name for the Jina sentence-transformer model.
        """
        super().__init__(path)
        self.model = SentenceTransformer(path, trust_remote_code=True)

    def get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            Tuple[List[List[float]], int]: A tuple containing the list of embeddings and the embedding dimension.
        """
        embeddings = self.model.encode(texts)
        if not texts or embeddings.size == 0:
            # Attempt to get dimension from model config if no texts are provided
            dim = self.model.get_sentence_embedding_dimension()
            return [], dim if dim is not None else 0

        dim = len(embeddings[0])
        return embeddings.tolist(), dim
