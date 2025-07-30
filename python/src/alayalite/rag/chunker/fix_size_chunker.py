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
This module provides the FixSizeChunker class for splitting text into fixed-size chunks.
"""

import os
import sys

from langchain_text_splitters import CharacterTextSplitter
from rag.chunker.base import BaseChunker

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)


class FixSizeChunker(BaseChunker):
    """
    A class for chunking text into fixed-size chunks with optional overlap.

    Attributes:
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping tokens between chunks.
        separator (str): The separator string to use for chunking (default is "\\n\\n").
        length_function (function): Function to calculate the length of a chunk (default is len).
    """

    def chunking(self, docs):
        """
        Splits a document into chunks of a fixed size.

        Args:
            docs (str): The document text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            is_separator_regex=False,
        )
        return text_splitter.split_text(docs)
