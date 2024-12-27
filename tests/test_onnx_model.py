import os
import unittest
from pathlib import Path

import numpy as np

from midi2hands.models.onnex.onnex_model import ONNXModel

onnex_path = Path(os.path.dirname(__file__)) / "resources/model.onnx"


class TestOnexModel(unittest.TestCase):
  def setUp(self) -> None:
    self.model = ONNXModel(onnx_path=onnex_path)

  def test_onnx_model(self):
    out = self.model(np.ones([1, 30, 3], dtype=np.float32))
    assert isinstance(out, list)
    assert isinstance(out[0], float)
