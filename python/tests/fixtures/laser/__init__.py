# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""LASER test fixture builder package (test-only).

Module-level imports are kept minimal so this package loads cleanly on
unsupported builds (Linux+OFF / macOS / Windows). The actual
`alayalite.laser` / `alayalite.vamana` imports happen lazily inside
`builder.build_small_laser_artifacts`.
"""
