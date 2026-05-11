# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides the base class for all text chunker implementations.
"""


class BaseChunker:
    def __init__(self, chunk_size, chunk_overlap, separator=" ", length_function=len):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.length_function = length_function

    def chunking(self, docs) -> list:
        raise NotImplementedError
