"""High-level reproducible workflows for composer-classification experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_pipeline import MANIFEST_PATH, build_clean_manifest, load_manifest, load_or_extract_metadata
from features import SplitFeatures, build_split_features
from models import CNNComposerClassifier, ComposerDataset, LSTMComposerClassifier, class_weights
from project_config import COMPOSERS, MODELS_DIR, SEED
from training import Evaluation, choose_device, train_lstm


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters shared by the LSTM baseline and its later CNN comparison."""

    max_sequence_length: int = 4096
    batch_size: int | None = None
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    regenerate_manifest: bool = False
    refresh_eda: bool = False


@dataclass
class ExperimentData:
    """Prepared data, EDA summaries, and DataLoaders for one fixed split."""

    device: torch.device
    manifest: pd.DataFrame
    exclusions: list[dict[str, str]]
    inventory: pd.DataFrame
    train_metadata: pd.DataFrame
    eda_summary: pd.DataFrame
    split_features: SplitFeatures
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader


@dataclass
class BaselineResult:
    """Artifacts from a validation-selected model baseline."""

    model: nn.Module
    history: dict[str, list[float]]
    validation: Evaluation
    class_weights: torch.Tensor


def seed_everything(seed: int = SEED) -> None:
    """Seed NumPy and PyTorch for repeatable local experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_experiment(config: ExperimentConfig = ExperimentConfig()) -> ExperimentData:
    """Build or load a manifest, load cached EDA, and create DataLoaders."""
    seed_everything()
    if config.regenerate_manifest or not MANIFEST_PATH.exists():
        manifest, exclusions = build_clean_manifest()
    else:
        manifest, exclusions = load_manifest(), []

    inventory = (
        manifest.groupby(["composer", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=COMPOSERS, columns=["train", "validation", "test"], fill_value=0)
    )
    inventory["total"] = inventory.sum(axis=1)

    metadata = load_or_extract_metadata(
        manifest,
        refresh=config.regenerate_manifest or config.refresh_eda,
    )
    train_metadata = metadata.query("split == 'train'").copy()
    eda_columns = [
        "duration_s",
        "tempo_bpm",
        "note_count",
        "note_density",
        "pitch_range",
        "mean_pitch",
    ]
    eda_summary = train_metadata.groupby("composer")[eda_columns].mean().round(2).reindex(COMPOSERS)

    split_features = build_split_features(manifest, max_length=config.max_sequence_length)
    device = choose_device()
    batch_size = config.batch_size or (16 if device.type == "mps" else 32)
    train_features, train_labels = split_features.train
    validation_features, validation_labels = split_features.validation
    test_features, test_labels = split_features.test

    return ExperimentData(
        device=device,
        manifest=manifest,
        exclusions=exclusions,
        inventory=inventory,
        train_metadata=train_metadata,
        eda_summary=eda_summary,
        split_features=split_features,
        train_loader=DataLoader(
            ComposerDataset(train_features, train_labels), batch_size=batch_size, shuffle=True
        ),
        validation_loader=DataLoader(
            ComposerDataset(validation_features, validation_labels), batch_size=batch_size
        ),
        test_loader=DataLoader(ComposerDataset(test_features, test_labels), batch_size=batch_size),
    )


def run_lstm_baseline(
    data: ExperimentData, config: ExperimentConfig = ExperimentConfig()
) -> BaselineResult:
    """Train the class-weighted LSTM baseline without evaluating the test set."""
    seed_everything()
    _, train_labels = data.split_features.train
    model = LSTMComposerClassifier(vocab_size=129, num_classes=len(COMPOSERS)).to(data.device)
    weights = class_weights(train_labels, len(COMPOSERS)).to(data.device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, validation = train_lstm(
        model=model,
        train_loader=data.train_loader,
        validation_loader=data.validation_loader,
        loss_function=criterion,
        optimizer=optimizer,
        device=data.device,
        epochs=config.epochs,
        checkpoint_path=MODELS_DIR / "best_lstm_model.pt",
        early_stopping_patience=config.early_stopping_patience,
    )
    return BaselineResult(
        model=model,
        history=history,
        validation=validation,
        class_weights=weights.detach().cpu(),
    )


def run_cnn_baseline(
    data: ExperimentData, config: ExperimentConfig = ExperimentConfig()
) -> BaselineResult:
    """Train the class-weighted 1D CNN baseline without evaluating the test set."""
    seed_everything()
    _, train_labels = data.split_features.train
    model = CNNComposerClassifier(vocab_size=129, num_classes=len(COMPOSERS)).to(data.device)
    weights = class_weights(train_labels, len(COMPOSERS)).to(data.device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, validation = train_lstm(
        model=model,
        train_loader=data.train_loader,
        validation_loader=data.validation_loader,
        loss_function=criterion,
        optimizer=optimizer,
        device=data.device,
        epochs=config.epochs,
        checkpoint_path=MODELS_DIR / "best_cnn_model.pt",
        early_stopping_patience=config.early_stopping_patience,
    )
    return BaselineResult(
        model=model,
        history=history,
        validation=validation,
        class_weights=weights.detach().cpu(),
    )