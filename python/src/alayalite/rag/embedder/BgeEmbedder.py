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

from typing import List, Tuple

from .base import BaseEmbedding


class BgeEmbedder(BaseEmbedding):
    def __init__(self, path: str = "BAAI/bge-m3") -> None:
        from FlagEmbedding import BGEM3FlagModel

        super().__init__(path)
        self.model = BGEM3FlagModel(model_name_or_path=self.path, use_fp16=False)

    def get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        embeddings = self.model.encode(texts, batch_size=1, max_length=8192)["dense_vecs"]
        dim = len(embeddings[0])
        return embeddings, dim
