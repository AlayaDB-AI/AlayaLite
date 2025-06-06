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
from typing import List

import numpy as np
from rag.chunker.base import BaseChunker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)


class SemanticChunker(BaseChunker):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.8, window_size: int = 3):
        """
        基于语义相似度的动态文本分块器

        参数:
            model_name: 语义编码模型名称，默认'all-MiniLM-L6-v2'
            threshold: 相似度阈值（0-1），默认0.8
            window_size: 滑动窗口大小，默认3个句子
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.window_size = window_size

    def chunking(self, text: str) -> List[str]:
        """
        实现语义感知的分块逻辑

        参数:
            text: 输入文本

        返回:
            List[str]: 分割后的文本块列表
        """
        # Step 1: 基础分句
        sentences = self._split_into_sentences(text)

        # Step 2: 计算句子嵌入
        embeddings = self._encode_sentences(sentences)

        # Step 3: 滑动窗口分析
        chunks = []
        current_chunk = []
        for i in range(len(sentences)):
            current_chunk.append(sentences[i])

            # 当积累足够句子时开始检测
            if len(current_chunk) >= self.window_size:
                # 比较当前窗口与下一个窗口的相似度
                if self._should_split(current_chunk, embeddings, i):
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []

        # 添加最后剩余的内容
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _should_split(self, current_chunk: List[str], embeddings: np.ndarray, current_index: int) -> bool:
        """
        使用滑动窗口计算语义相似度判断是否需要分割

        计算逻辑：
        previous_window = [sent_{k-n}, ..., sent_k]
        next_window = [sent_{k+1}, ..., sent_{k+n+1}]
        similarity = cos_sim(mean(prev_window), mean(next_window))
        """
        window_size = self.window_size

        # 获取当前窗口和下一窗口的索引范围
        prev_start = max(0, current_index - window_size + 1)
        next_end = min(len(embeddings) - 1, current_index + window_size)

        # 计算平均嵌入
        prev_emb = np.mean(embeddings[prev_start : current_index + 1], axis=0)
        next_emb = np.mean(embeddings[current_index + 1 : next_end + 1], axis=0)

        # 计算余弦相似度
        similarity = cosine_similarity([prev_emb], [next_emb])[0][0]
        return similarity < self.threshold

    def _encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """批量编码句子为嵌入向量"""
        return self.model.encode(sentences, convert_to_numpy=True)

    def _split_into_sentences(self, text: str) -> List[str]:
        """基础分句实现（可根据需要替换更复杂的分句逻辑）"""
        return [s.strip() for s in text.split(".") if s.strip()]
