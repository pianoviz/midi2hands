from typing import Any

import numpy as np
from numpy.typing import NDArray

from midi2hands.models.interface import HandModel


class ModelMock(HandModel):
  # def __init__(self, window_size: int = 30):
  #   self.window_size = window_size

  def __call__(self, x: NDArray[np.float32]) -> list[float]:
    return [0.3] * x.shape[0]

  @property
  def model(self) -> Any:
    return None

  @property
  def window_size(self) -> int:
    return 30
