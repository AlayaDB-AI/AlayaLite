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

import numpy as np
from alayalite import Client, Index

from ..base.module import BaseANN


class AlayaLite(BaseANN):
    def __init__(self, metric, param):
        self.index_save_dir = "/home/app/results/alaya_indices"
        self.client = Client(self.index_save_dir)
        self.index = None
        self.ef = None
        self.save_index_name = "rabitq_index_rabitq_test"

        self.index_type = param["index_type"]
        self.metric = metric
        self.quantization_type = param["quantization_type"]
        self.fit_threads = param["fit_threads"]
        self.search_threads = param["search_threads"]
        self.R = param["R"]
        self.L = param["L"]
        self.M = param["M"]

        print("alaya init done")

    def fit(self, X: np.array) -> None:
        if os.path.exists(os.path.join(self.index_save_dir, self.save_index_name)):
            self.index = Index.load(self.index_save_dir, self.save_index_name)
            print("load index from cache")
        else:
            X = X.astype(np.float32)
            self.index = self.client.create_index(
                name=self.save_index_name,
                metric=self.metric,
                quantization_type=self.quantization_type,
                capacity=X.shape[0],
            )
            self.index.fit(vectors=X, num_threads=self.fit_threads)
            self.client.save_index(self.save_index_name)
            print("save index to cache")

    def set_query_arguments(self, ef):
        self.ef = int(ef)

    def prepare_query(self, q: np.array, n: int):
        self.q = q
        self.n = n

    def run_prepared_query(self):
        self.res = self.index.search(query=self.q, topk=self.n, ef_search=self.ef)

    def batch_query(self, X: np.array, n: int) -> None:
        self.res = self.index.batch_search(queries=X, topk=n, ef_search=self.ef)

    def get_prepared_query_results(self):
        return self.res

    def get_batch_results(self) -> np.array:
        return self.res

    def __str__(self) -> str:
        return "AlayaDB_Lite"
