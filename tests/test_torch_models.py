import unittest
from pathlib import Path

import numpy as np

from midi2hands.config import Config
from midi2hands.models.torch.lstm import LSTMModel
from midi2hands.models.torch.transformer import TransformerModel


class TestTorchModels(unittest.TestCase):
  def setUp(self):
    # self.config = Config(model="lstm")
    self.lstm_config = Config(model="lstm", device="mps", input_size=3)
    self.lstm_model = LSTMModel(config=self.lstm_config)

    self.transformer_config = Config(
      model="transformer",
      device="mps",
      input_size=4,
      hidden_size=32,
      num_layers=2,
      dropout=0.1,
    )
    self.transformer_model = TransformerModel(config=self.transformer_config)

  def test_lstm_model_return_type(self):
    ret = self.lstm_model(np.ones((1, self.lstm_config.window_size, self.lstm_config.input_size), dtype=np.float32))
    assert isinstance(ret, list), f"Expected type 'list', got {type(ret)}"
    assert all(isinstance(item, float) for item in ret), "Not all elements in the list are of type 'float'"

  def test_transformer_model_return_type(self):
    ret = self.transformer_model(np.ones((1, self.transformer_config.window_size, self.transformer_config.input_size), dtype=np.float32))
    assert isinstance(ret, list), f"Expected type 'list', got {type(ret)}"
    assert all(isinstance(item, float) for item in ret), "Not all elements in the list are of type 'float'"

  def test_to_onnx(self):
    output_path = Path("/tmp")
    self.lstm_model.to_onnx(output_path=output_path)
