# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides the BgeEmbedder class for creating embeddings using BAAI's BGE models.
"""

from typing import List, Tuple

from FlagEmbedding import BGEM3FlagModel

from .base import BaseEmbedding


class BgeEmbedder(BaseEmbedding):
    """An embedding class that uses BAAI's BGE sentence-transformer models."""

    def __init__(self, path: str = "BAAI/bge-m3") -> None:
        """
        Initializes the BgeEmbedder.

        Args:
            path (str): The model path or name for the BGE model.
        """
        super().__init__(path)
        # For bge-m3, it is recommended to use BGEM3FlagModel.
        self.model = BGEM3FlagModel(model_name_or_path=self.path, use_fp16=False)

    def get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            Tuple[List[List[float]], int]: A tuple containing the list of embeddings and the embedding dimension.
        """
        if not texts:
            # Attempt to get dimension from model config if no texts are provided
            try:
                dim = self.model.model.config.hidden_size
            except AttributeError:
                dim = 0  # Fallback if config is not available
            return [], dim

        # Note: BGE-M3 model's encode function returns a dictionary.
        embeddings = self.model.encode(texts, batch_size=1, max_length=8192)["dense_vecs"]
        dim = len(embeddings[0])
        return embeddings.tolist(), dim
