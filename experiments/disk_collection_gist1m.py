# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""This is a compatibility wrapper around `python -m alayalite.bench.disk_collection`.

See openspec change `disk-collection-benchmark-suite` for full context."""

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("--n", "1000000"), ("--queries", "1000"), ("--k", "10"), ("--out", "results")):
        parser.add_argument(name, default=default)
    parser.add_argument("--dataset-root", "--data-dir", dest="dataset_root", required=True)
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "-m",
        "alayalite.bench.disk_collection",
        "--engine",
        "disk_flat",
        "--dataset",
        "gist1m",
        "--n",
        args.n,
        "--queries",
        args.queries,
        "--k",
        args.k,
        "--out",
        args.out,
        "--dataset-root",
        args.dataset_root,
    ]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
