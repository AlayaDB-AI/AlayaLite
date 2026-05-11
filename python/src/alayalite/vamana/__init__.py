# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""AlayaLite Vamana graph builder.

The binding is compiled into ``alayalite._alayalitepy.vamana`` as a
submodule. The module exposes a single function, ``build_index``, that
writes a DiskANN-format ``.index`` file from a ``.fbin`` dataset.

The CLI ``build_vamana_index`` (under ``tools/build_vamana_index/``) and
this Python binding share one dispatch library
(``include/index/graph/vamana/build_dispatch.hpp``). Given identical
parameters and ``num_threads=1``, both produce byte-for-byte identical
outputs — see ``tests/vamana/test_cli_vs_python_parity.py``.
"""

from __future__ import annotations

from alayalite._alayalitepy import vamana as _vamana_mod  # type: ignore[attr-defined]

build_index = _vamana_mod.build_index

__all__ = ["build_index"]
