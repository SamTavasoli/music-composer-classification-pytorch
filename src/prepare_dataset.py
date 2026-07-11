"""Prepare the four-composer MIDI dataset for the AAI-511 final project.

This script downloads the Kaggle "midi-classic-music" dataset (Blanderbuss, 2025),
extracts the four required composers (Bach, Beethoven, Chopin, Mozart), and
creates train/dev/test splits and writes a manifest file with split counts.

Usage:
    cd final-project/music-composer-classification-pytorch
    python src/prepare_dataset.py

Source:
    Blanderbuss. (2025). MIDI Classic Music [Data set]. Kaggle.
    https://www.kaggle.com/datasets/blanderbuss/midi-classic-music
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
KAGGLE_DATASET = "blanderbuss/midi-classic-music"
SOURCE_DIR_NAME = "midiclassics"
TARGET_DIR_NAME = "midi_classic_music_filtered"

TARGET_COMPOSERS = ["Bach", "Beethoven", "Chopin", "Mozart"]
SEED = 511

# Default split proportions (train / dev / test)
SPLIT_PROPORTIONS = (0.70, 0.15, 0.15)


def find_midi_files(root: Path) -> list[Path]:
    """Recursively collect all MIDI files under *root*."""
    midi_files: list[Path] = []
    for ext in ("*.mid", "*.MID", "*.midi", "*.MIDI"):
        midi_files.extend(root.rglob(ext))
    # De-duplicate by resolved path and ignore macOS resource forks.
    unique: dict[str, Path] = {}
    for p in midi_files:
        if ".__" in p.name or p.name.startswith("._"):
            continue
        unique[str(p.resolve())] = p
    return sorted(unique.values())


def split_files(
    files: list[Path],
    proportions: tuple[float, float, float] = SPLIT_PROPORTIONS,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Deterministic train/dev/test split."""
    n = len(files)
    if n < 3:
        raise ValueError(f"Need at least 3 files for a split, got {n}")

    rng = np.random.default_rng(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = max(1, int(round(n * proportions[0])))
    n_dev = max(1, int(round(n * proportions[1])))
    # Avoid rounding edge cases leaving test empty.
    n_test = n - n_train - n_dev
    if n_test < 1:
        n_test = 1
        n_train = max(1, n - n_dev - n_test)

    train_idx = indices[:n_train].tolist()
    dev_idx = indices[n_train : n_train + n_dev].tolist()
    test_idx = indices[n_train + n_dev :].tolist()

    files_arr = np.array(files)
    return (
        files_arr[train_idx].tolist(),
        files_arr[dev_idx].tolist(),
        files_arr[test_idx].tolist(),
    )


def copy_files(src_paths: Iterable[Path], dst_dir: Path) -> None:
    """Copy or hard-link files into *dst_dir*, flattening subdirectories.

    Hard links are used when source and destination reside on the same file
    system, which is much faster and avoids duplicating the MIDI data.
    """
    import re

    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_paths:
        # Sanitize characters that can break os.link on some filesystems
        # (e.g., single quotes used in Kaggle filenames).
        safe_name = re.sub(r'[<>"|?*]', '_', src.name)
        dst = dst_dir / safe_name
        counter = 1
        stem = src.stem
        suffix = src.suffix
        while dst.exists():
            dst = dst_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        if src.resolve() == dst.resolve():
            continue
        try:
            os.link(src, dst)
        except OSError:
            # Different file system, link not supported, or other link error.
            shutil.copy2(src, dst)


def download_kaggle_dataset(download_dir: Path) -> None:
    """Download and unzip the Kaggle dataset if it is not already present."""
    source_dir = download_dir / SOURCE_DIR_NAME
    if source_dir.exists() and any(source_dir.iterdir()):
        print(f"Source directory already exists: {source_dir}")
        return

    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET,
                "-p",
                str(download_dir),
                "--unzip",
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI not found. Install with: pip install kaggle "
            "and configure ~/.kaggle/kaggle.json."
        ) from exc

    if not source_dir.exists():
        raise RuntimeError(
            f"Expected source directory not found after download: {source_dir}"
        )


def prepare_dataset(
    dataset_root: Path = DATASET_ROOT,
    output_root: Path | str | None = None,
) -> dict[str, dict[str, int]]:
    """Create the filtered, split dataset.

    Args:
        dataset_root: Root directory containing the raw Kaggle download.
        output_root: Directory where the filtered dataset will be written.
            Defaults to <dataset_root>/midi_classic_music_filtered.

    Returns:
        A nested dict mapping composer -> split -> file count.
    """
    source_root = dataset_root / SOURCE_DIR_NAME
    target_root = Path(output_root) if output_root else dataset_root / TARGET_DIR_NAME

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    # Clean previous target.
    if target_root.exists():
        shutil.rmtree(target_root)

    inventory: dict[str, dict[str, int]] = {}

    for composer in TARGET_COMPOSERS:
        composer_dir = source_root / composer
        if not composer_dir.exists():
            raise FileNotFoundError(f"Composer directory missing: {composer_dir}")

        files = find_midi_files(composer_dir)
        if not files:
            raise RuntimeError(f"No MIDI files found for {composer}")

        train_files, dev_files, test_files = split_files(files)

        for split, split_files_list in [
            ("train", train_files),
            ("dev", dev_files),
            ("test", test_files),
        ]:
            split_dir = target_root / split / composer
            copy_files(split_files_list, split_dir)
            inventory.setdefault(composer, {})[split] = len(split_files_list)

    # Write a small manifest for reproducibility.
    manifest = {
        "source": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        "composers": TARGET_COMPOSERS,
        "seed": SEED,
        "split_proportions": SPLIT_PROPORTIONS,
        "inventory": inventory,
    }
    manifest_path = target_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return inventory


def print_inventory(inventory: dict[str, dict[str, int]]) -> None:
    """Pretty-print the dataset inventory."""
    print("\nDataset inventory:")
    print("-" * 50)
    total = 0
    for composer in TARGET_COMPOSERS:
        counts = inventory[composer]
        comp_total = sum(counts.values())
        total += comp_total
        print(
            f"{composer:12}  train: {counts['train']:4}  "
            f"dev: {counts['dev']:4}  test: {counts['test']:4}  total: {comp_total:4}"
        )
    print("-" * 50)
    print(f"{'Total':12} {'':>19} {'':>7} {total:>9}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the four-composer MIDI dataset from Kaggle."
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DATASET_ROOT,
        help="Directory where the raw Kaggle dataset lives or will be downloaded.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume the raw dataset is already present; do not call Kaggle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the filtered dataset will be written. Defaults to "
             "<download-dir>/midi_classic_music_filtered.",
    )
    args = parser.parse_args(argv)

    # When invoked from the project repo (e.g., via the notebook), keep the
    # filtered dataset inside the project-level dataset folder so it is found
    # by portable notebooks.
    download_dir = args.download_dir.resolve()
    output_dir = (args.output_dir or download_dir / TARGET_DIR_NAME).resolve()

    if not args.skip_download:
        download_kaggle_dataset(download_dir)

    inventory = prepare_dataset(download_dir, output_dir)
    print_inventory(inventory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
