from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from midi2hands.models.interface import HandModel


class ONNXModel(HandModel):
  def __init__(self, window_size: int, onnx_path: Path):
    self.window_size = window_size
    self.session = ort.InferenceSession(str(onnx_path))
    # You might need to figure out input/output names for your ONNX graph
    self.input_name: str = self.session.get_inputs()[0].name  # type: ignore
    self.output_name: str = self.session.get_outputs()[0].name  # type: ignore

  def __call__(self, x: NDArray[np.float32]) -> list[float]:
    # Convert x to a suitable shape or type if needed
    ort_inputs = {self.input_name: x}
    ort_outs = self.session.run([self.output_name], ort_inputs)  # type: ignore
    # Convert to Python floats
    return ort_outs[0].flatten().tolist()  # type: ignore

  @property
  def model(self) -> Any:
    # Not a PyTorch model, so might just return None or the session
    return self.session

  def load_model(self, model_path: Path, device: Literal["cpu", "gpu", "mps"]) -> None:
    # Possibly re-create or update the session if needed
    self.session = ort.InferenceSession(str(model_path))

  def save_model(self, model_path: Path) -> None:
    # ONNX models are typically exported from PyTorch or other frameworks,
    # so there's no "training" inside ONNX runtime. Possibly just copy if needed.
    pass
