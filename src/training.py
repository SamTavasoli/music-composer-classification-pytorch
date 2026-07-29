"""Reproducible training and validation functions for PyTorch classifiers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class Evaluation:
    """Aggregate and per-class validation metrics."""

    loss: float
    accuracy: float
    report: dict[str, dict[str, float]]
    labels: np.ndarray
    predictions: np.ndarray


def choose_device() -> torch.device:
    """Select an available hardware accelerator, then fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, loss_function: nn.Module, device: torch.device) -> Evaluation:
    """Evaluate a model without retaining gradients."""
    model.eval()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for pitch, continuous, labels in loader:
            pitch, continuous, labels = pitch.to(device), continuous.to(device), labels.to(device)
            logits = model(pitch, continuous)
            total_loss += loss_function(logits, labels).item() * labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    return Evaluation(
        loss=total_loss / len(loader.dataset),
        accuracy=float((target_array == prediction_array).mean()),
        report=classification_report(target_array, prediction_array, output_dict=True, zero_division=0),
        labels=target_array,
        predictions=prediction_array,
    )


def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    early_stopping_patience: int | None = 5,
) -> tuple[dict[str, list[float]], Evaluation]:
    """Train with validation checkpointing and early stopping on validation performance.

    Checkpoint selection and the early-stopping counter both use validation
    accuracy (validation loss as a tie-break), so training halts once the model
    has gone ``early_stopping_patience`` epochs without a new best checkpoint.
    This directly targets the overfitting pattern observed in earlier runs,
    where later epochs kept lowering train loss while validation loss diverged.
    """
    history = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}
    best_evaluation: Evaluation | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for pitch, continuous, labels in train_loader:
            pitch, continuous, labels = pitch.to(device), continuous.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pitch, continuous)
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        validation = evaluate(model, validation_loader, loss_function, device)
        history["train_loss"].append(total_loss / total)
        history["train_accuracy"].append(correct / total)
        history["validation_loss"].append(validation.loss)
        history["validation_accuracy"].append(validation.accuracy)
        print(
            f"Epoch {epoch:02d}/{epochs} | train loss {history['train_loss'][-1]:.4f} | "
            f"train accuracy {history['train_accuracy'][-1]:.4f} | "
            f"validation loss {validation.loss:.4f} | validation accuracy {validation.accuracy:.4f}"
        )
        if best_evaluation is None or (validation.accuracy, -validation.loss) > (
            best_evaluation.accuracy,
            -best_evaluation.loss,
        ):
            best_evaluation = validation
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"epoch": epoch, "model_state_dict": best_state}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch:02d}: no validation improvement in "
                    f"{early_stopping_patience} epochs (best epoch {best_epoch:02d})."
                )
                break

    if best_state is None or best_evaluation is None:
        raise RuntimeError("Training did not process any batches.")
    model.load_state_dict(best_state)
    return history, best_evaluation