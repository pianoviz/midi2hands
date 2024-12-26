from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from midi2hands.models.interface import HandModel


class ModelMock(HandModel):
  def __init__(self, window_size: int = 30):
    self.window_size = window_size

  def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([0.3] * x.shape[0])

  @property
  def model(self) -> Any:
    return None

  def save_model(self, model_path: Path) -> None:
    return None

  def load_model(self, model_path: Path, device: Literal["cpu", "gpu", "mps"]) -> None:
    return None
