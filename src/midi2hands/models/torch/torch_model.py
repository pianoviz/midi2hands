from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from midi2hands.config import Config
from midi2hands.models.interface import HandModel


class TorchModel(HandModel):
  _model: torch.nn.Module
  config: Config

  def __call__(self, x: NDArray[np.float32]) -> list[float]:
    """Calculate"""
    outputs = self._model(torch.Tensor(x).to(self.config.device))
    outputs.squeeze()
    y_pred = [float(item[0]) for item in outputs.cpu().detach().numpy().tolist()]
    return y_pred

  def save_model(self, model_path: Path) -> None:
    torch.save(self._model.state_dict(), model_path)  # type: ignore

  def load_model(self, model_path: Path, device: Literal["cpu", "gpu", "mps"] = "cpu") -> None:
    state_dict = torch.load(model_path, map_location=torch.device(device))  # type: ignore
    self._model.load_state_dict(state_dict)

  def to_onnx(self, output_path: Path) -> None:
    self._model.eval()
    torch.onnx.export(  # type: ignore
      self._model,
      (torch.randn(1, self.config.window_size, self.config.input_size).to(self.config.device),),
      Path(output_path) / "model.onnx",
    )
    print(f"Model successfully exported to ONNX format at {output_path}")
