"""MIDI sequence extraction and training-only feature normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

from project_config import COMPOSERS, RAW_DATA_DIR


FEATURE_NAMES = ("pitch", "velocity", "duration", "time_interval")


@dataclass
class SplitFeatures:
    """Feature matrices and labels partitioned by the file-level manifest."""

    train: tuple[np.ndarray, np.ndarray]
    validation: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]
    label_map: dict[str, int]
    normalization: dict[str, tuple[float, float]]


def midi_to_sequence(path: Path, max_length: int) -> np.ndarray:
    """Create a padded pitch, velocity, duration, and interval matrix for one MIDI file."""
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = sorted(
        (note for instrument in midi.instruments for note in instrument.notes),
        key=lambda note: (note.start, note.pitch, note.end),
    )
    if not notes:
        raise ValueError(f"MIDI file has no notes: {path}")

    sequence = np.zeros((max_length, len(FEATURE_NAMES)), dtype=np.float32)
    previous_start = 0.0
    for index, note in enumerate(notes[:max_length]):
        sequence[index] = (
            note.pitch + 1,
            note.velocity,
            max(0.0, note.end - note.start),
            max(0.0, note.start - previous_start) if index else 0.0,
        )
        previous_start = note.start
    return sequence


def _normalize_splits(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    normalized = [values.copy() for values in (train, validation, test)]
    train_mask = train[:, :, 0] != 0
    statistics: dict[str, tuple[float, float]] = {}
    for feature_index, feature_name in enumerate(FEATURE_NAMES[1:], start=1):
        values = train[:, :, feature_index][train_mask]
        mean = float(values.mean())
        std = float(values.std()) or 1.0
        statistics[feature_name] = (mean, std)
        for array in normalized:
            mask = array[:, :, 0] != 0
            array[:, :, feature_index][mask] = (array[:, :, feature_index][mask] - mean) / std
            array[:, :, feature_index][~mask] = 0.0
    return (*normalized, statistics)


def build_split_features(
    manifest: pd.DataFrame,
    max_length: int = 4096,
    raw_root: Path = RAW_DATA_DIR / "midiclassics",
) -> SplitFeatures:
    """Extract all manifest files and normalize continuous features from training data only."""
    label_map = {composer: index for index, composer in enumerate(COMPOSERS)}
    extracted: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    labels: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for row in manifest.itertuples(index=False):
        if row.split not in extracted:
            raise ValueError(f"Unknown split: {row.split}")
        extracted[row.split].append(midi_to_sequence(raw_root / row.relative_path, max_length))
        labels[row.split].append(label_map[row.composer])

    arrays = {split: np.stack(values) for split, values in extracted.items()}
    train, validation, test, statistics = _normalize_splits(
        arrays["train"], arrays["validation"], arrays["test"]
    )
    return SplitFeatures(
        train=(train, np.asarray(labels["train"], dtype=np.int64)),
        validation=(validation, np.asarray(labels["validation"], dtype=np.int64)),
        test=(test, np.asarray(labels["test"], dtype=np.int64)),
        label_map=label_map,
        normalization=statistics,
    )