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

import os
import sys

from rag.chunker.base import BaseChunker

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)


class SentenceChunker(BaseChunker):
    """
    A class for chunking text into fixed-size chunks with optional overlap.

    Attributes:
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping sentences between chunks.
        separator (str): The separator string to use for chunking (default is "\n\n").
        length_function (function): Function to calculate the length of a chunk (default is len).
    """

    def __init__(self, chunk_size, chunk_overlap, separator=" ", length_function=len):
        super().__init__(chunk_size, chunk_overlap)

    def chunking(self, docs):
        import re

        chunks = []

        # 使用正则表达式按句子结束符分割文档
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|。|？|！)\s*", docs)
        sentences = [s.strip() for s in sentences if s.strip()]  # 去除空白和空字符串

        start = 0
        while start < len(sentences):
            # 计算当前块的结束位置
            end = min(start + self.chunk_size, len(sentences))

            # 获取当前块
            current_chunk = sentences[start:end]
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # 更新起始位置以处理重叠
            start = start + self.chunk_size - self.chunk_overlap

        return chunks
