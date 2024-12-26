import copy
import logging
import random
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np
import torch
from joblib import Memory
from numpy._typing import NDArray

# from mido.midifiles.midifiles import MidiFile
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from midi2hands.config import Config

torch.manual_seed(0)  # type: ignore


memory = Memory(location="cache", verbose=0)


class MidiDataset(Dataset[Any]):
  def __init__(self, windows: NDArray[np.float32], labels: NDArray[np.float32]):
    self.windows = windows
    self.labels = labels

  def __len__(self):
    return len(self.windows)

  def __getitem__(self, idx: int):
    return self.windows[idx], self.labels[idx]


def get_device() -> Literal["cpu", "cuda", "mps"]:
  return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def convert_hand_to_number(hand: str | None):
  return 0 if hand == "left" else (1 if hand == "right" else -1)


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
  return float(np.mean([np.array(y_true) == np.array(y_pred)]))


def setup_logger(name: str, logdir: str, filename: str) -> logging.Logger:
  path = Path(logdir)
  if not path.exists():
    path.mkdir(parents=True)
  path = path / filename
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  file_handler = logging.FileHandler(path)
  file_handler.setLevel(logging.INFO)
  formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)  # add the file handler
  return logger


def generate_complex_random_name():
  adjectives = [
    "nostalgic",
    "wonderful",
    "mystic",
    "quiet",
    "vibrant",
    "eager",
    "frosty",
    "peaceful",
    "serene",
    "ancient",
  ]
  nouns = [
    "morse",
    "turing",
    "neumann",
    "lovelace",
    "hopper",
    "tesla",
    "einstein",
    "bohr",
    "darwin",
    "curie",
  ]
  numbers = range(10, 99)  # two digit numbers

  adjective = random.choice(adjectives)
  noun = random.choice(nouns)
  number = random.choice(numbers)
  return f"{adjective}_{noun}_{number}"


def k_fold_split(k: int) -> list[Tuple[list[Path], list[Path]]]:
  paths = list(Path("data/train").rglob("*.mid"))
  n = len(paths)
  fold_size = n // k
  folds: list[Tuple[list[Path], list[Path]]] = []
  for i in range(k):
    start = i * fold_size
    end = (i + 1) * fold_size
    val_fold = paths[start:end]
    train_fold = paths[:start] + paths[end:]
    folds.append((train_fold, val_fold))
  return folds


def process_batch(
  windows: torch.Tensor,
  labels: torch.Tensor,
  model: torch.nn.Module,
  criterion: torch.nn.Module,
  device: str,
) -> tuple[torch.nn.Module, list[int], list[int]]:
  windows = windows.to(device)
  labels = labels.float().to(device)
  outputs = model(windows)

  loss = criterion(outputs, labels)

  # Simplify handling of batches with single sample
  labels = labels.squeeze()
  outputs = outputs.squeeze()
  if labels.ndim == 0:
    labels = labels.unsqueeze(0)
  if outputs.ndim == 0:
    outputs = outputs.unsqueeze(0)

  y_true: list[int] = labels.cpu().numpy().astype(int).tolist()  # type: ignore
  y_pred: list[int] = np.where(outputs.cpu().detach().numpy() < 0.5, 0, 1).tolist()

  return loss, y_true, y_pred


def train_loop(
  model: torch.nn.Module,
  train_loader: DataLoader[Any],
  val_loader: DataLoader[Any],
  optimizer: Optimizer,
  criterion: torch.nn.Module,
  logger: logging.Logger,
  config: Config,
):
  # Store metrics for all epochs
  metrics: dict[str, list[float]] = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
  }

  num_epochs = config.num_epochs
  if config.use_early_stopping:
    num_epochs = 500

  # Function to process batches

  best_val_score = 0
  best_model_state = None
  epochs_without_improvement = 0

  for epoch in range(num_epochs):
    model.train()
    train_losses: list[float] = []
    train_accs: list[float] = []
    for windows, labels in train_loader:
      loss, y_t, y_p = process_batch(windows, labels, model, criterion, config.device)
      train_losses.append(loss.item())
      train_accs.append(accuracy(y_t, y_p))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Logging training metrics
    metrics["train_loss"].append(float(np.mean(train_losses)))
    metrics["train_acc"].append(float(np.mean(train_accs)))

    # Validation phase
    model.eval()
    val_losses: list[float] = []
    val_accs: list[float] = []
    with torch.no_grad():
      for windows, labels in val_loader:
        loss, y_t, y_p = process_batch(windows, labels, model, criterion, config.device)
        val_losses.append(loss.item())
        val_accs.append(accuracy(y_t, y_p))

    # Logging validation metrics
    metrics["val_loss"].append(float(np.mean(val_losses)))
    metrics["val_acc"].append(float(np.mean(val_accs)))

    # Output log for each epoch
    logger.info(
      f"Epoch {epoch+1:02}\ttrain_loss: {metrics['train_loss'][-1]:.4f}\t"
      f"train_acc: {metrics['train_acc'][-1]:.4f}\t"
      f"val_loss: {metrics['val_loss'][-1]:.4f}\t"
      f"val_acc: {metrics['val_acc'][-1]:.4f}"
    )

    # Early stopping
    if config.use_early_stopping:
      if metrics["val_acc"][-1] > best_val_score:
        best_val_score = metrics["val_acc"][-1]
        epochs_without_improvement = 0
        best_model_state = copy.deepcopy(model.state_dict())
      else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
          logger.info(f"Early stopping after {epoch+1} epochs without improvement")
          break
      # Load the best model state
  if best_model_state is not None:
    logger.info("Loading best model state")
    model.load_state_dict(best_model_state)
  return metrics
