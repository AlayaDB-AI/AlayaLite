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

# 忽略与 "subnormal numbers" 相关的警告
import warnings

warnings.filterwarnings(
    "ignore",  # 隐藏警告
    message="The value of the smallest subnormal for <class 'numpy.float32'> type is zero.",
)
warnings.filterwarnings(
    "ignore", message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero."
)

from .client import Client  # noqa: E402
from .collection import Collection  # noqa: E402
from .index import Index  # noqa: E402
from .utils import calc_gt, calc_recall, load_fvecs, load_ivecs  # noqa: E402

__all__ = [
    "Client",
    "Index",
    "Collection",
    # utils
    "load_fvecs",
    "load_ivecs",
    "calc_recall",
    "calc_gt",
]

__version__ = "0.1.0"
