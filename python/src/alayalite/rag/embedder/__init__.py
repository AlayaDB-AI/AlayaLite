# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This package provides various text embedding models and a factory function
to access them easily.
"""

# RAG/Embedder/__init__.py

from .bge_embedder import BgeEmbedder
from .embedder import embedder, get_embedder
from .jina_embedder import JinaEmbedder
from .m3e_embedder import M3eEmbedder
from .multilingual_embedder import MultilingualEmbedder

__all__ = [
    "BgeEmbedder",
    "M3eEmbedder",
    "MultilingualEmbedder",
    "JinaEmbedder",
    "embedder",
    "get_embedder",
]
