#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Refuse commits that introduce Chinese characters (or other CJK) in source.

Walks every file passed via sys.argv and flags any line containing a CJK
Unified Ideograph (BMP block U+4E00–U+9FFF, Extension A U+3400–U+4DBF, or
Extensions B–F U+20000–U+2EBEF). Comments, docstrings, strings, and code
are all checked indiscriminately — translate or anglicize before committing.

When to run:
- Automatically as a pre-commit hook (see .pre-commit-config.yaml)
- Manually: `python scripts/ci/check_chinese.py <path1> <path2> ...`

Exit code:
    0  no Chinese characters found
    1  at least one match (offending file:line:content printed)
"""

import re
import sys


def main() -> int:
    """Check files for Chinese characters."""
    # CJK Unified Ideographs + Extension A + Extension B-F
    pattern = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\U00020000-\U0002EBEF]")
    found = False

    for filepath in sys.argv[1:]:
        try:
            with open(filepath, encoding="utf-8", errors="surrogateescape") as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.search(line):
                        print(f"{filepath}:{line_num}: {line.rstrip()}")
                        found = True
        except OSError as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)
            continue

    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
