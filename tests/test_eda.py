import pandas as pd
import pytest

from eda import EDA_FEATURES, analyze_training_metadata


def test_analyze_training_metadata_returns_pca_and_correlations() -> None:
    rows = []
    for index in range(8):
        row = {feature: float((feature_index + 1) * (index + 1)) for feature_index, feature in enumerate(EDA_FEATURES)}
        row["composer"] = "Bach" if index < 4 else "Mozart"
        rows.append(row)

    analysis = analyze_training_metadata(pd.DataFrame(rows))

    assert analysis.pca_scores.shape == (8, 3)
    assert analysis.correlation.shape == (len(EDA_FEATURES), len(EDA_FEATURES))
    assert sum(analysis.explained_variance_ratio) == pytest.approx(1.0)


def test_analyze_training_metadata_rejects_missing_features() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        analyze_training_metadata(pd.DataFrame({"composer": ["Bach"]}))