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
This module provides the M3eEmbedder class for creating embeddings using M3E models.
"""

from typing import List, Tuple

from sentence_transformers import SentenceTransformer

from .base import BaseEmbedding


class M3eEmbedder(BaseEmbedding):
    """An embedding class that uses M3E sentence-transformer models."""

    def __init__(self, path: str = "moka-ai/m3e-base") -> None:
        """
        Initializes the M3eEmbedder.

        Args:
            path (str): The model path or name for the M3E sentence-transformer model.
        """
        super().__init__(path)
        self.model = SentenceTransformer(path)

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
            dim = self.model.get_sentence_embedding_dimension()
            return [], dim if dim is not None else 0

        embeddings = self.model.encode(texts)
        dim = len(embeddings[0])
        return embeddings.tolist(), dim
