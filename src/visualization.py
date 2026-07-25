"""Plots for EDA and model evaluation in the master notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from project_config import COMPOSERS, FIGURES_DIR
from eda import EDAAnalysis
from training import Evaluation


def plot_exploration(
    inventory: pd.DataFrame,
    train_metadata: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[plt.Figure, plt.Figure]:
    """Create and save split-inventory and training-only feature plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    inventory_figure, inventory_axis = plt.subplots(figsize=(8, 4))
    inventory[["train", "validation", "test"]].plot(
        kind="bar", stacked=True, ax=inventory_axis
    )
    inventory_axis.set(title="Clean MIDI File Distribution by Split", xlabel="Composer", ylabel="Files")
    inventory_axis.tick_params(axis="x", rotation=0)
    inventory_figure.tight_layout()
    inventory_figure.savefig(figures_dir / "class_distribution.png", dpi=150)

    feature_figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.boxplot(data=train_metadata, x="composer", y="note_density", ax=axes[0])
    axes[0].set_title("Training-Split Note Density")
    sns.boxplot(data=train_metadata, x="composer", y="pitch_range", ax=axes[1])
    axes[1].set_title("Training-Split Pitch Range")
    feature_figure.tight_layout()
    feature_figure.savefig(figures_dir / "eda_note_density_pitch_range.png", dpi=150)
    return inventory_figure, feature_figure


def plot_eda_analysis(
    analysis: EDAAnalysis,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[plt.Figure, plt.Figure]:
    """Create and save training-split PCA and correlation plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    pca_figure, pca_axis = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=analysis.pca_scores,
        x="PC1",
        y="PC2",
        hue="composer",
        alpha=0.7,
        ax=pca_axis,
    )
    first, second = analysis.explained_variance_ratio
    pca_axis.set(
        title="Training-Split PCA of Standardized EDA Features",
        xlabel=f"PC1 ({first:.1%} variance)",
        ylabel=f"PC2 ({second:.1%} variance)",
    )
    pca_figure.tight_layout()
    pca_figure.savefig(figures_dir / "eda_pca.png", dpi=150)

    correlation_figure, correlation_axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        analysis.correlation,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        ax=correlation_axis,
    )
    correlation_axis.set_title("Training-Split EDA Feature Correlations")
    correlation_figure.tight_layout()
    correlation_figure.savefig(figures_dir / "eda_correlation.png", dpi=150)
    return pca_figure, correlation_figure


def plot_lstm_results(
    history: dict[str, list[float]],
    evaluation: Evaluation,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[plt.Figure, plt.Figure]:
    """Create and save loss-history and validation-confusion-matrix plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    history_figure, history_axis = plt.subplots(figsize=(9, 4))
    history_frame[["train_loss", "validation_loss"]].plot(ax=history_axis)
    history_axis.set(title="LSTM Loss", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    history_figure.tight_layout()
    history_figure.savefig(figures_dir / "lstm_loss.png", dpi=150)

    matrix = confusion_matrix(evaluation.labels, evaluation.predictions)
    matrix_figure, matrix_axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=COMPOSERS).plot(
        cmap="Blues", colorbar=False, ax=matrix_axis
    )
    matrix_axis.set_title(f"LSTM Validation Confusion Matrix, Accuracy = {evaluation.accuracy:.3f}")
    matrix_figure.tight_layout()
    matrix_figure.savefig(figures_dir / "lstm_validation_confusion_matrix.png", dpi=150)
    return history_figure, matrix_figure


def plot_cnn_results(
    history: dict[str, list[float]],
    evaluation: Evaluation,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[plt.Figure, plt.Figure]:
    """Create and save CNN loss-history and validation-confusion-matrix plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    history_figure, history_axis = plt.subplots(figsize=(9, 4))
    history_frame[["train_loss", "validation_loss"]].plot(ax=history_axis)
    history_axis.set(title="CNN Loss", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    history_figure.tight_layout()
    history_figure.savefig(figures_dir / "cnn_loss.png", dpi=150)

    matrix = confusion_matrix(evaluation.labels, evaluation.predictions)
    matrix_figure, matrix_axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=COMPOSERS).plot(
        cmap="Blues", colorbar=False, ax=matrix_axis
    )
    matrix_axis.set_title(f"CNN Validation Confusion Matrix, Accuracy = {evaluation.accuracy:.3f}")
    matrix_figure.tight_layout()
    matrix_figure.savefig(figures_dir / "cnn_validation_confusion_matrix.png", dpi=150)
    return history_figure, matrix_figure