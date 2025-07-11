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

from .BgeEmbedder import BgeEmbedder
from .JinaEmbedder import JinaEmbedder
from .M3eEmbedder import M3eEmbedder
from .MultilingualEmbedder import MultilingualEmbedder


def get_embedder(model_name: str = "bge-m3", model_path: str = "") -> Tuple[List[List[float]], int]:
    if model_name.startswith("bge"):
        if model_path:
            embedder_instance = BgeEmbedder(path=model_path)
        else:
            embedder_instance = BgeEmbedder()
    elif model_name.startswith("m3e"):
        if model_path:
            embedder_instance = M3eEmbedder(path=model_path)
        else:
            embedder_instance = M3eEmbedder()
    elif model_name.startswith("multilingual"):
        if model_path:
            embedder_instance = MultilingualEmbedder(path=model_path)
        else:
            embedder_instance = MultilingualEmbedder()
    elif model_name.startswith("jina"):
        if model_path:
            embedder_instance = JinaEmbedder(path=model_path)
        else:
            embedder_instance = JinaEmbedder()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return embedder_instance


def embedder(texts: List[str], model_name: str = "bge-m3", path: str = "") -> Tuple[List[List[float]], int]:
    if model_name.startswith("bge"):
        if path:
            embedder_instance = BgeEmbedder(path=path)
        else:
            embedder_instance = BgeEmbedder()
    elif model_name.startswith("m3e"):
        if path:
            embedder_instance = M3eEmbedder(path=path)
        else:
            embedder_instance = M3eEmbedder()
    elif model_name.startswith("multilingual"):
        if path:
            embedder_instance = MultilingualEmbedder(path=path)
        else:
            embedder_instance = MultilingualEmbedder()
    elif model_name.startswith("jina"):
        if path:
            embedder_instance = JinaEmbedder(path=path)
        else:
            embedder_instance = JinaEmbedder()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return embedder_instance.get_embeddings(texts)
