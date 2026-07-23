#!/usr/bin/env python3
"""
Download and extract Qdrant filtered benchmark datasets.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

DATASETS = {
    "qdrant_random_ints_1m": {
        "dirname": "qdrant-random-ints-1m",
        "archive": "random_ints_1m.tgz",
        "url": "https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_ints_1m.tgz",
    }
}

REQUIRED_FILES = ("vectors.npy", "payloads.jsonl", "tests.jsonl")


def find_dataset_root(path: Path) -> Path:
    if all((path / filename).exists() for filename in REQUIRED_FILES):
        return path

    matches = [candidate.parent for candidate in path.rglob("vectors.npy") if candidate.is_file()]
    matches = [
        candidate for candidate in matches if all((candidate / filename).exists() for filename in REQUIRED_FILES)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Unable to locate extracted dataset root under {path}")
    return matches[0]


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if member_path != destination and destination not in member_path.parents:
            raise RuntimeError(f"Refusing to extract archive member outside destination: {member.name}")
    archive.extractall(destination)


def materialize_dataset(dataset_key: str, output_dir: Path) -> Path:
    if dataset_key not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_key}")

    dataset = DATASETS[dataset_key]
    dataset_dir = output_dir / dataset["dirname"]
    if all((dataset_dir / filename).exists() for filename in REQUIRED_FILES):
        return dataset_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alayalite-qdrant-dataset-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        archive_path = temp_dir / dataset["archive"]
        extract_dir = temp_dir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {dataset['url']} -> {archive_path}")
        download_file(dataset["url"], archive_path)

        print(f"Extracting {archive_path} -> {extract_dir}")
        with tarfile.open(archive_path, "r:gz") as archive:
            safe_extract(archive, extract_dir)

        extracted_root = find_dataset_root(extract_dir)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        shutil.copytree(extracted_root, dataset_dir)

    return dataset_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Qdrant filtered ANN benchmark dataset")
    parser.add_argument("--dataset", default="qdrant_random_ints_1m", choices=sorted(DATASETS))
    parser.add_argument("--output-dir", required=True, help="Parent directory that will contain the extracted dataset")
    args = parser.parse_args()

    dataset_dir = materialize_dataset(args.dataset, Path(args.output_dir).expanduser().resolve())
    print(dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
