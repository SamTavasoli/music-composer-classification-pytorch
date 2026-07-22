"""Dataset inventory, cleaning, and reproducible split-manifest utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pretty_midi
from mido.midifiles.meta import KeySignatureError
from sklearn.model_selection import train_test_split

from project_config import COMPOSERS, PROCESSED_DATA_DIR, RAW_DATA_DIR, SEED


MIDI_SUFFIXES = {".mid", ".midi"}
MANIFEST_PATH = PROCESSED_DATA_DIR / "dataset_split_manifest.csv"
SUMMARY_PATH = PROCESSED_DATA_DIR / "dataset_manifest.json"
EDA_METADATA_PATH = PROCESSED_DATA_DIR / "eda_metadata.csv"


def find_midi_files(raw_root: Path = RAW_DATA_DIR / "midiclassics") -> list[tuple[str, Path]]:
    """Return MIDI files for the configured composers in deterministic order."""
    records: list[tuple[str, Path]] = []
    for composer in COMPOSERS:
        composer_dir = raw_root / composer
        if not composer_dir.exists():
            raise FileNotFoundError(f"Composer directory is missing: {composer_dir}")
        for path in sorted(composer_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in MIDI_SUFFIXES and not path.name.startswith("._"):
                records.append((composer, path))
    return records


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_clean_manifest(
    raw_root: Path = RAW_DATA_DIR / "midiclassics",
    manifest_path: Path = MANIFEST_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Parse valid, unique files and create a stratified 70/15/15 split manifest."""
    valid_records: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    seen_hashes: set[str] = set()

    for composer, path in find_midi_files(raw_root):
        relative_path = path.relative_to(raw_root)
        file_hash = _file_hash(path)
        if file_hash in seen_hashes:
            errors.append({"file": str(relative_path), "error": "exact duplicate"})
            continue
        try:
            pretty_midi.PrettyMIDI(str(path))
        except (OSError, ValueError, KeySignatureError) as exc:
            errors.append({"file": str(relative_path), "error": str(exc)})
            continue
        seen_hashes.add(file_hash)
        valid_records.append({"composer": composer, "relative_path": str(relative_path)})

    frame = pd.DataFrame(valid_records)
    if frame.empty:
        raise RuntimeError("No valid MIDI files were found after data-quality checks.")

    train, temporary = train_test_split(
        frame, test_size=0.30, stratify=frame["composer"], random_state=SEED
    )
    validation, test = train_test_split(
        temporary, test_size=0.50, stratify=temporary["composer"], random_state=SEED
    )
    split_frames = []
    for split_name, split_frame in (("train", train), ("validation", validation), ("test", test)):
        assigned = split_frame.copy()
        assigned["split"] = split_name
        split_frames.append(assigned)
    manifest = pd.concat(split_frames, ignore_index=True).sort_values(
        ["split", "composer", "relative_path"]
    ).reset_index(drop=True)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "raw_root": str(raw_root),
        "seed": SEED,
        "valid_files": len(manifest),
        "excluded_files": len(errors),
        "inventory": manifest.groupby(["composer", "split"]).size().unstack(fill_value=0).to_dict(),
        "errors": errors,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest, errors


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Load a previously generated manifest and validate its required columns."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_path}. Run build_clean_manifest first."
        )
    manifest = pd.read_csv(manifest_path)
    required_columns = {"composer", "relative_path", "split"}
    if not required_columns.issubset(manifest.columns):
        raise ValueError(f"Manifest must include {sorted(required_columns)}.")
    return manifest


def extract_metadata(
    manifest: pd.DataFrame,
    raw_root: Path = RAW_DATA_DIR / "midiclassics",
) -> pd.DataFrame:
    """Extract file-level musical summary features for EDA from manifest records."""
    records: list[dict[str, float | int | str]] = []
    for row in manifest.itertuples(index=False):
        midi = pretty_midi.PrettyMIDI(str(raw_root / row.relative_path))
        notes = [note for instrument in midi.instruments for note in instrument.notes]
        if not notes:
            continue
        pitches = [note.pitch for note in notes]
        tempos = midi.get_tempo_changes()[1]
        duration = midi.get_end_time()
        records.append(
            {
                "composer": row.composer,
                "split": row.split,
                "duration_s": duration,
                "tempo_bpm": float(tempos.mean()) if len(tempos) else 0.0,
                "note_count": len(notes),
                "note_density": len(notes) / duration if duration else 0.0,
                "pitch_range": max(pitches) - min(pitches),
                "mean_pitch": float(sum(pitches) / len(pitches)),
            }
        )
    return pd.DataFrame(records)


def load_or_extract_metadata(
    manifest: pd.DataFrame,
    refresh: bool = False,
    metadata_path: Path = EDA_METADATA_PATH,
) -> pd.DataFrame:
    """Load cached EDA metadata or parse MIDI files once and cache the result.

    Rebuild the cache when a new manifest is intentionally created or when the
    caller requests ``refresh=True`` after changing the EDA feature definition.
    """
    if metadata_path.exists() and not refresh:
        return pd.read_csv(metadata_path)

    metadata = extract_metadata(manifest)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    return metadata