"""Utilities for monitoring process memory usage."""

import psutil


def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024 / 1024
