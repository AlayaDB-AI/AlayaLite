"""Laser on-disk Quantized Graph index.

The binding is compiled into ``alayalite._alayalitepy.laser`` as a
submodule (Phase L3 unified layout). If the wheel was built with
``-DALAYA_ENABLE_LASER=OFF`` the import below raises ``ImportError`` —
reinstall with Laser enabled to use this package.
"""

from __future__ import annotations

from alayalite._alayalitepy import laser as _laser_mod  # type: ignore[attr-defined]

Index = _laser_mod.Index

__all__ = ["Index"]
