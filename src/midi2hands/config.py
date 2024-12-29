import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict


class Device(Enum):
  CPU = "cpu"
  CUDA = "cuda"
  MPS = "mps"


class BaseConfig(ABC):
  @property
  @abstractmethod
  def device(self) -> Device:
    pass

  @property
  @abstractmethod
  def window_size(self) -> int:
    pass

  @property
  @abstractmethod
  def input_size(self) -> int:
    pass

  @abstractmethod
  def to_dict(self) -> Dict[str, Any]:
    pass

  @classmethod
  def from_json(cls, json_path: Path) -> "BaseConfig":
    with open(json_path, "r") as f:
      data = json.load(f)
    return cls(**data)

  def __repr__(self) -> str:
    attrs = ", ".join(f"{key}={value}" for key, value in vars(self).items())
    return f"{self.__class__.__name__}({attrs})"


class TrainingConfig:
  def __init__(
    self,
    batch_size: int = 64,
    num_epochs: int = 2,
    seed: int = 64,
    use_early_stopping: bool = True,
    patience: int = 5,
    use_kfold: bool = False,
    n_folds: int = 10,
    device: Device = Device.CPU,
    inference_eval: bool = True,
  ) -> None:
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.seed = seed
    self.use_early_stopping = use_early_stopping
    self.patience = patience
    self.use_kfold = use_kfold
    self.n_folds = n_folds
    self.device = device
    self.inference_eval = inference_eval

  def to_dict(self) -> Dict[str, Any]:
    return vars(self)

  @classmethod
  def from_json(cls, json_path: Path) -> "TrainingConfig":
    with open(json_path, "r") as f:
      data = json.load(f)
    return cls(**data)

  def __repr__(self) -> str:
    attrs = ", ".join(f"{key}={value}" for key, value in vars(self).items())
    return f"{self.__class__.__name__}({attrs})"


class LSTMConfig(BaseConfig):
  def __init__(
    self,
    device: Device = Device.CPU,
    window_size: int = 30,
    input_size: int = 4,
    hidden_size: int = 32,
    dropout: float = 0.1,
    num_layers: int = 3,
    num_classes: int = 1,
  ) -> None:
    self._device = device
    self._window_size = window_size
    self._input_size = input_size
    self.hidden_size = hidden_size
    self.dropout = dropout
    self.num_layers = num_layers
    self.num_classes = num_classes

  @property
  def device(self) -> Device:
    return self._device

  @property
  def window_size(self) -> int:
    return self._window_size

  @property
  def input_size(self) -> int:
    return self._input_size

  def to_dict(self) -> Dict[str, Any]:
    return vars(self)


class TransformerConfig(BaseConfig):
  def __init__(
    self,
    device: Device = Device.CPU,
    window_size: int = 30,
    input_size: int = 3,
    hidden_size: int = 32,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
    num_classes: int = 1,
    dim_feedforward: int = 64,
  ) -> None:
    self._device = device
    self._window_size = window_size
    self._input_size = input_size
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.dropout = dropout
    self.num_classes = num_classes
    self.dim_feedforward = dim_feedforward

  @property
  def device(self) -> Device:
    return self._device

  @property
  def window_size(self) -> int:
    return self._window_size

  @property
  def input_size(self) -> int:
    return self._input_size

  def to_dict(self) -> Dict[str, Any]:
    return vars(self)
