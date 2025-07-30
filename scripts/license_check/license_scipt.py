import argparse
import os
import sys
from pathlib import Path

# Apache License Headers
LICENSE_HEADER_PY = """\
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
"""

LICENSE_HEADER_CPP = """\
/*
 * Copyright 2025 AlayaDB.AI
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
"""

EXTENSIONS_PY = {".py"}
EXTENSIONS_CPP = {".c", ".cpp", ".h", ".hpp"}


def has_license(content):
    return "Licensed under the Apache License" in content


def check_license_file(file_path: Path, header_type: str):
    with file_path.open("r", encoding="utf-8") as f:
        content = f.read()
    return has_license(content)


def add_license_to_file(file_path: Path, header: str):
    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    content = "".join(lines)
    if has_license(content):
        print(f"Unchanged: {file_path}")
        return False
    shebang = ""
    body_start = 0
    if lines and lines[0].startswith("#!"):
        shebang = lines[0]
        body_start = 1
    new_content = (shebang if shebang else "") + header + "\n" + "".join(lines[body_start:])
    with file_path.open("w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Added license to: {file_path}")
    return True


def process_directory(directory: Path, check_only=False, missing_files=None):
    modified = 0
    unchanged = 0
    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext in EXTENSIONS_PY:
            has_it = check_license_file(file_path, "py")
            if check_only:
                if not has_it:
                    print(f"Missing license: {file_path}")
                    if missing_files is not None:
                        missing_files.append(str(file_path))
                else:
                    print(f"Checked OK: {file_path}")
            else:
                if add_license_to_file(file_path, LICENSE_HEADER_PY):
                    modified += 1
                else:
                    unchanged += 1
        elif ext in EXTENSIONS_CPP:
            has_it = check_license_file(file_path, "cpp")
            if check_only:
                if not has_it:
                    print(f"Missing license: {file_path}")
                    if missing_files is not None:
                        missing_files.append(str(file_path))
                else:
                    print(f"Checked OK: {file_path}")
            else:
                if add_license_to_file(file_path, LICENSE_HEADER_CPP):
                    modified += 1
                else:
                    unchanged += 1
    if not check_only:
        return modified, unchanged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add/Check License Header for source files.")
    parser.add_argument(
        "-c",
        "--check-only",
        action="store_true",
        help="Only check if license header exists, do not modify files (CI mode)",
    )
    args = parser.parse_args()

    roots = [
        Path("./python"),
        Path("./include"),
        Path("./tests"),
        Path("./examples"),
    ]
    if args.check_only:
        all_missing = []
        for root in roots:
            if root.exists():
                process_directory(root, check_only=True, missing_files=all_missing)
            else:
                print(f"Directory not found: {root}")
        print("\n=== License Header Check Summary ===")
        print(f"Total files missing license: {len(all_missing)}")
        if all_missing:
            sys.exit(1)
        else:
            print("All checked files have license header.")
            sys.exit(0)
    else:
        total_modified = 0
        total_unchanged = 0
        for root in roots:
            if root.exists():
                m, u = process_directory(root)
                total_modified += m
                total_unchanged += u
            else:
                print(f"Directory not found: {root}")
        print("\n=== License Header Summary ===")
        print(f"Modified files:   {total_modified}")
        print(f"Unchanged files:  {total_unchanged}")
