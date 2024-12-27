import torch
from torch import nn

from midi2hands.config import LSTMConfig
from midi2hands.models.torch.torch_model import TorchModel


class LSTMModel(TorchModel):
  def __init__(self, config: LSTMConfig):
    self.config = config
    self._model = LSTMModule(config).to(config.device)

  @property
  def model(self) -> torch.nn.Module:
    return self._model

  @property
  def window_size(self) -> int:
    return self.config.window_size


class LSTMModule(nn.Module):
  def __init__(self, config: LSTMConfig):
    super(LSTMModule, self).__init__()  # type: ignore
    self.hidden_size = config.hidden_size
    self.num_layers = config.num_layers
    self.device = config.device
    self.lstm = nn.LSTM(
      config.input_size,
      config.hidden_size,
      config.num_layers,
      batch_first=True,
      bidirectional=True,
      dropout=config.dropout,
    )
    self.fc = nn.Linear(config.hidden_size * 2, 10)
    self.fc2 = nn.Linear(10, config.num_classes)

  def forward(self, x: torch.Tensor):
    h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
    c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
    # print(x.shape, h0.shape, c0.shape)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, out.size(1) // 2, :])
    out = self.fc2(out)
    out = torch.sigmoid(out)
    return out
