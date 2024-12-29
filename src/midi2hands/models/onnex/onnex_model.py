import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from midi2hands.models.interface import HandModel

default_path = Path(os.path.join(os.path.dirname(__file__), "model.onnx"))


class ONNXModel(HandModel):
  def __init__(self, onnx_path: Path = default_path):
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

  @property
  def window_size(self) -> int:
    """The onnx model is exported with a sample input."""
    # the input sample has shape (bs, window_size, num_features)
    return self.session.get_inputs()[0].shape[1]  # type: ignore
