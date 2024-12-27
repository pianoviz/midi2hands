import torch
from torch import nn

from midi2hands.config import Config
from midi2hands.models.torch.torch_model import TorchModel


class LSTMModel(TorchModel):
  def __init__(self, config: Config):
    self.config = config
    self._model = LSTMModule(device=config.device, input_size=config.input_size).to(config.device)

  @property
  def model(self) -> torch.nn.Module:
    return self._model

  @property
  def window_size(self) -> int:
    return self.config.window_size


class LSTMModule(nn.Module):
  def __init__(
    self,
    input_size: int = 4,
    hidden_size: int = 32,
    num_layers: int = 3,
    num_classes: int = 1,
    device: str = "cpu",
    dropout: float = 0.1,
  ):
    super(LSTMModule, self).__init__()  # type: ignore
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.device = device
    self.lstm = nn.LSTM(
      input_size,
      hidden_size,
      num_layers,
      batch_first=True,
      bidirectional=True,
      dropout=dropout,
    )
    self.fc = nn.Linear(hidden_size * 2, 10)
    self.fc2 = nn.Linear(10, num_classes)

  def forward(self, x: torch.Tensor):
    h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
    c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
    # print(x.shape, h0.shape, c0.shape)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, out.size(1) // 2, :])
    out = self.fc2(out)
    out = torch.sigmoid(out)
    return out
