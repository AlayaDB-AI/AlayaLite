# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides factory functions for creating and using various text embedding models.
"""

from typing import List, Tuple

from .base import BaseEmbedding
from .bge_embedder import BgeEmbedder
from .jina_embedder import JinaEmbedder
from .m3e_embedder import M3eEmbedder
from .multilingual_embedder import MultilingualEmbedder


def get_embedder(model_name: str = "bge-m3", model_path: str = "") -> BaseEmbedding:
    """
    Factory function to get an embedder instance based on the model name.

    Args:
        model_name (str): The name of the model to use (e.g., 'bge', 'm3e').
        model_path (str): Optional path to a local model. If not provided, downloads the default model.

    Returns:
        BaseEmbedding: An instance of the corresponding embedder class.
    """
    model_map = {
        "bge": BgeEmbedder,
        "m3e": M3eEmbedder,
        "multilingual": MultilingualEmbedder,
        "jina": JinaEmbedder,
    }

    for key, embedder_class in model_map.items():
        if model_name.startswith(key):
            return embedder_class(path=model_path) if model_path else embedder_class()

    raise ValueError(f"Unsupported model: {model_name}")


def embedder(texts: List[str], model_name: str = "bge-m3", path: str = "") -> Tuple[List[List[float]], int]:
    """
    Creates an embedder and uses it to generate embeddings for a list of texts.

    Args:
        texts (List[str]): The list of texts to embed.
        model_name (str): The name of the model to use.
        path (str): Optional path to a local model.

    Returns:
        Tuple[List[List[float]], int]: A tuple containing the embeddings and their dimension.
    """
    embedder_instance = get_embedder(model_name=model_name, model_path=path)
    return embedder_instance.get_embeddings(texts)
