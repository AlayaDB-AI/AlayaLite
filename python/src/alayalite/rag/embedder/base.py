# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides the base class for all text embedding model implementations.
"""

from typing import List, Tuple


class BaseEmbedding:
    """Abstract base class for embedding models."""

    def __init__(self, path: str) -> None:
        """
        Initializes the base embedding model.

        Args:
            path (str): The path or name of the embedding model.
        """
        self.path = path

    def get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generates embeddings for a list of texts. This method must be implemented by subclasses.
        """
        raise NotImplementedError
