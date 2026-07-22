"""PyTorch datasets and model definitions for composer classification."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class ComposerDataset(Dataset):
    """Provide pitch tokens, continuous note features, and class labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.pitch = torch.as_tensor(features[:, :, 0], dtype=torch.long)
        self.continuous_features = torch.as_tensor(features[:, :, 1:], dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pitch[index], self.continuous_features[index], self.labels[index]


class LSTMComposerClassifier(nn.Module):
    """Classify a variable-length MIDI note sequence using its final LSTM state."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        continuous_feature_count: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 4,
        dropout_rate: float = 0.30,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim + continuous_feature_count,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pitch: torch.Tensor, continuous_features: torch.Tensor) -> torch.Tensor:
        lengths = (pitch != 0).sum(dim=1).clamp(min=1)
        combined = torch.cat((self.embedding(pitch), continuous_features), dim=2)
        outputs, _ = self.lstm(combined)
        last_note_indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(2))
        final_states = outputs.gather(1, last_note_indices).squeeze(1)
        return self.classifier(self.dropout(final_states))


class CNNComposerClassifier(nn.Module):
    """Classify note sequences with local 1D convolutional pattern detectors."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        continuous_feature_count: int = 3,
        num_filters: int = 64,
        num_classes: int = 4,
        dropout_rate: float = 0.30,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        input_channels = embedding_dim + continuous_feature_count
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(input_channels, num_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(num_filters * 2, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_filters * 2, num_classes)

    def forward(self, pitch: torch.Tensor, continuous_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((self.embedding(pitch), continuous_features), dim=2)
        features = combined.transpose(1, 2)
        features = self.conv_block_1(features)
        features = self.conv_block_2(features)
        features = self.conv_block_3(features)
        pooled_features = self.global_pool(features).squeeze(-1)
        return self.classifier(self.dropout(pooled_features))


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Calculate inverse-frequency class weights for cross-entropy loss."""
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("Every class must appear in the training split.")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)