# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
AlayaLite Python SDK.

This is the main entry point for the alayalite package, providing easy access
to all key components like the Client, Collection, and utility functions.
"""

import warnings

from ._alayalitepy import DiskCollection, MetricType  # noqa: E402
from .client import Client  # noqa: E402
from .collection import Collection  # noqa: E402
from .index import Index  # noqa: E402
from .utils import calc_gt, calc_recall, load_fvecs, load_ivecs  # noqa: E402

# Ignore warnings related to "subnormal numbers"
warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal for <class 'numpy.float32'> type is zero.",
)
warnings.filterwarnings(
    "ignore", message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero."
)


__all__ = [
    "Client",
    "Index",
    "Collection",
    "DiskCollection",
    "MetricType",
    # utils
    "load_fvecs",
    "load_ivecs",
    "calc_recall",
    "calc_gt",
]

__version__ = "0.1.0"
