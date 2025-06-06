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
from dataclasses import dataclass, field

import numpy as np

from alayalite import Client, IndexParams

from ..base.module import BaseANN


class AlayaLite(BaseANN):
    def __init__(self, metric, param):
        self.params = AlayaParams(metric=metric, **param)
        self.client = Client()
        self.index = None
        self.ef = None
        print(self.params)

        print("alaya init done")

    def fit(self, X: np.array) -> None:
        self.index = self.client.create_index(
            params=IndexParams(
                index_type=self.params.index_type,
                quantization_type=self.params.quantization_type,
                metric=self.params.metric,
                capacity=X.shape[0],  # auto expand capacity
            )
        )

        self.index.fit(
            vectors=X,
            M=self.params.M,
            R=self.params.R,
            L=self.params.L,
            num_threads=self.params.fit_threads,
        )

    def set_query_arguments(self, ef):
        self.ef = int(ef)

    def prepare_query(self, q: np.array, n: int):
        self.q = q
        self.n = n

    def run_prepared_query(self):
        self.res = self.index.search(query=self.q, topk=self.n, ef_search=self.ef)

    def batch_query(self, X: np.array, n: int) -> None:
        self.res = self.index.batch_search(queries=X, topk=n, ef_search=self.ef, num_threads=self.params.search_threads)

    def get_prepared_query_results(self):
        return self.res

    def get_batch_results(self) -> np.array:
        return self.res

    def __str__(self) -> str:
        return "AlayaDB_Lite"


@dataclass
class AlayaParams:
    M: int
    R: int
    L: int
    index_type: str
    metric: str
    capacity: np.uint32 = field(default=100000)
    quantization_type: str = field(default="none")
    fit_threads: int = field(default=os.cpu_count())
    search_threads: int = field(default=os.cpu_count())
    index_save_dir: str = field(default="/home/app/results/alaya_indices")

    def __post_init__(self):
        self.M = int(self.M)
        self.R = int(self.R)
        self.L = int(self.L)
        self.index_type = str(self.index_type)
        self.metric = str(self.metric)
        self.capacity = np.uint32(self.capacity)
        self.quantization_type = str(self.quantization_type)
        self.fit_threads = int(self.fit_threads)
        self.search_threads = int(self.search_threads)
        self.index_save_dir = str(self.index_save_dir)
