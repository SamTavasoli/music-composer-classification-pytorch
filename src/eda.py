"""Training-split exploratory analysis for composer metadata."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


EDA_FEATURES = (
    "duration_s",
    "tempo_bpm",
    "note_count",
    "note_density",
    "pitch_range",
    "mean_pitch",
)


@dataclass(frozen=True)
class EDAAnalysis:
    """Correlation and two-component PCA results for training metadata."""

    correlation: pd.DataFrame
    pca_scores: pd.DataFrame
    explained_variance_ratio: tuple[float, float]


def analyze_training_metadata(metadata: pd.DataFrame) -> EDAAnalysis:
    """Compute standardized PCA and raw-feature correlations on training rows."""
    missing = set(EDA_FEATURES).difference(metadata.columns)
    if missing:
        raise ValueError(f"EDA metadata is missing columns: {sorted(missing)}")
    if "composer" not in metadata.columns:
        raise ValueError("EDA metadata is missing the composer column.")

    features = metadata.loc[:, EDA_FEATURES]
    standardized = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(standardized)
    pca_scores = pd.DataFrame(scores, columns=["PC1", "PC2"], index=metadata.index)
    pca_scores["composer"] = metadata["composer"].to_numpy()

    explained = tuple(float(value) for value in pca.explained_variance_ratio_)
    return EDAAnalysis(
        correlation=features.corr(),
        pca_scores=pca_scores,
        explained_variance_ratio=(explained[0], explained[1]),
    )