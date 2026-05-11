# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides factory functions to create and use various text chunkers.
"""

import os
import sys

from rag.chunker.FixSizeChunker import FixSizeChunker
from rag.chunker.SemanticChunker import SemanticChunker
from rag.chunker.SentenceChunker import SentenceChunker

# Add the parent directory to the system path to allow for package-level imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)


def get_chunker(chunk_model="fix_size", chunksize=3, overlap=0, semantic_model="all-MiniLM-L6-v2"):
    """
    Factory function to get a chunker instance based on the specified model.
    """
    if chunk_model == "fix_size":
        chunker_instance = FixSizeChunker(chunk_size=chunksize, chunk_overlap=overlap)
    elif chunk_model == "semantic":
        chunker_instance = SemanticChunker(window_size=chunksize, model_name=semantic_model)
    elif chunk_model == "sentence":
        chunker_instance = SentenceChunker(chunk_size=chunksize, chunk_overlap=overlap)
    else:
        raise ValueError(f"Unsupported chunk model: {chunk_model}")
    return chunker_instance


def chunker(docs, chunk_model="fix_size", chunksize=3, overlap=0, semantic_model="all-MiniLM-L6-v2"):
    """
    Creates a chunker and applies it to the given documents.
    """
    chunker_instance = get_chunker(chunk_model, chunksize, overlap, semantic_model)
    print(docs)
    return chunker_instance.chunking(docs)


# Example usage:
#
# sample_text = "Your sample text goes here..."
# chunks = chunker(sample_text, 'fix_size', 512, 50)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
#
# chunks = chunker(sample_text, 'semantic', 3)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
#
# chunks = chunker(sample_text, 'sentence', 10, 2)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
