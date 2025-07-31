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
# limitations under the license.

"""
This module provides the SentenceChunker class for splitting text into sentence-based chunks.
"""

import os
import re
import sys

from rag.chunker.base import BaseChunker

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)


class SentenceChunker(BaseChunker):
    """
    A class for chunking text into sentence-based chunks with optional overlap.

    Attributes:
        chunk_size (int): The maximum number of sentences per chunk.
        chunk_overlap (int): The number of overlapping sentences between chunks.
    """

    def chunking(self, docs):
        """
        Splits a document into chunks of sentences.

        Args:
            docs (str): The document text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        chunks = []

        # Split the document by sentence terminators
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|。|？|！)\s*", docs)
        sentences = [s.strip() for s in sentences if s.strip()]  # Remove whitespace and empty strings

        start_index = 0
        while start_index < len(sentences):
            # Calculate the end position of the current chunk
            end_index = min(start_index + self.chunk_size, len(sentences))

            # Get the current chunk
            current_chunk_sentences = sentences[start_index:end_index]
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)

            # Update the starting position to handle overlap
            start_index += self.chunk_size - self.chunk_overlap

        return chunks
