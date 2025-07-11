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

from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbedding


class MultilingualEmbedder(BaseEmbedding):
    def __init__(self, path: str = "intfloat/multilingual-e5-large") -> None:
        super().__init__(path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModel.from_pretrained(self.path)

    def get_embeddings(self, texts: List[str], model: str = "text-embedding-3-large") -> Tuple[List[List[float]], int]:
        batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**batch_dict)
        attention_mask = batch_dict["attention_mask"]
        last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        dim = len(embeddings[0])
        return embeddings, dim
