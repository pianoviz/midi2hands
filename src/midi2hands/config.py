import json
from pathlib import Path
from typing import Any, Dict, Literal


class BaseConfig:
  def __init__(self, seed: int = 42, device: Literal["cpu", "cuda", "mps"] = "cpu", window_size: int = 30, input_size: int = 4) -> None:
    self.seed = seed
    self.device = device
    self.window_size = window_size
    self.input_size = input_size

  @classmethod
  def from_json(cls, json_path: Path) -> "BaseConfig":
    with open(json_path, "r") as f:
      data: Dict[str, Any] = json.load(f)
    return cls(**data)

  def __repr__(self) -> str:
    attrs = ", ".join(f"{key}={value}" for key, value in vars(self).items())
    return f"{self.__class__.__name__}({attrs})"


class TrainingConfig:
  def __init__(
    self,
    batch_size: int = 64,
    num_epochs: int = 2,
    use_early_stopping: bool = True,
    patience: int = 5,
    use_kfold: bool = False,
    n_folds: int = 5,
    device: str = "cpu",
  ) -> None:
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.use_early_stopping = use_early_stopping
    self.patience = patience
    self.use_kfold = use_kfold
    self.n_folds = n_folds
    self.device = device

  @classmethod
  def from_json(cls, json_path: Path) -> "TrainingConfig":
    with open(json_path, "r") as f:
      data: Dict[str, Any] = json.load(f)
    return cls(**data)

  def __repr__(self) -> str:
    attrs = ", ".join(f"{key}={value}" for key, value in vars(self).items())
    return f"{self.__class__.__name__}({attrs})"


class LSTMConfig(BaseConfig):
  def __init__(
    self,
    seed: int = 42,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    input_size: int = 3,
    hidden_size: int = 32,
    dropout: float = 0.1,
    num_layers: int = 3,
    num_classes: int = 1,
  ) -> None:
    super().__init__(seed, device)
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.dropout = dropout
    self.num_layers = num_layers
    self.num_classes = num_classes


class TransformerConfig(BaseConfig):
  def __init__(
    self,
    seed: int = 42,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    input_size: int = 3,
    hidden_size: int = 32,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
    num_classes: int = 1,
    dim_feedforward: int = 64,
  ) -> None:
    super().__init__(seed, device)
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.dropout = dropout
    self.num_classes = num_classes
    self.dim_feedforward = dim_feedforward
