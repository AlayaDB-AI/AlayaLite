#!/usr/bin/env python3
"""Check for Chinese characters in source code files."""

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
