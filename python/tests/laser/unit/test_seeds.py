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

"""Tests for LASER seed derivation utilities."""

from __future__ import annotations

from alayalite.laser._seeds import derive_subseeds


def test_derive_subseeds_is_stable_for_same_master_seed() -> None:
    a = derive_subseeds(42)
    b = derive_subseeds(42)
    assert a == b


def test_derive_subseeds_do_not_overlap_for_different_master_seeds() -> None:
    a = derive_subseeds(1)
    b = derive_subseeds(2)
    assert set(a).isdisjoint(set(b))
