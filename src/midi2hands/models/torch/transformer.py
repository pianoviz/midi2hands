import torch
from torch import nn

from midi2hands.config import Config
from midi2hands.models.torch.torch_model import TorchModel


class TransformerModel(TorchModel):
  def __init__(self, config: Config):
    pass
    self.config = config
    self._model = TransformerModule(
      input_size=config.input_size, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout
    ).to(config.device)

  @property
  def model(self) -> torch.nn.Module:
    return self._model


class TransformerModule(nn.Module):
  def __init__(
    self,
    input_size: int = 4,
    hidden_size: int = 32,
    num_heads: int = 16,
    num_layers: int = 2,
    dim_feedforward: int = 64,
    dropout: float = 0.1,
  ):
    super(TransformerModule, self).__init__()  # type: ignore
    self.embedding = nn.Linear(input_size, hidden_size)
    self.transformer = nn.Transformer(
      d_model=hidden_size,
      nhead=num_heads,
      num_encoder_layers=num_layers,
      num_decoder_layers=num_layers,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      batch_first=True,
    )
    self.fc_out = nn.Linear(hidden_size, 1)  # Output layer with one unit for binary classification

  def forward(self, src: torch.Tensor):
    src = self.embedding(src)  # Embed the source sequence
    src = src.permute(1, 0, 2)  # Convert (batch_size, seq_len, embed_dim) to (seq_len, bs, embed_dim)
    transformer_output = self.transformer(src, src)
    # print(transformer_output.shape)
    output = self.fc_out(transformer_output[transformer_output.shape[0] // 2, :, :])
    # print(output.shape)
    output = torch.sigmoid(output)
    return output  # Convert back to (batch_size, seq_len, 1)
