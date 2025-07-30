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
This package provides various text chunking strategies, including fixed-size,
sentence-based, and semantic chunking.
"""

from .chunker import chunker, get_chunker
from .fix_size_chunker import FixSizeChunker
from .semantic_chunker import SemanticChunker
from .sentence_chunker import SentenceChunker

__all__ = ["FixSizeChunker", "SemanticChunker", "SentenceChunker", "chunker", "get_chunker"]
