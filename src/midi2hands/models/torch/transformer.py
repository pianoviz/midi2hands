import torch
from torch import nn

from midi2hands.config import TransformerConfig
from midi2hands.models.torch.torch_model import TorchModel


class TransformerModel(TorchModel):
  def __init__(self, config: TransformerConfig):
    pass
    self.config = config
    self._model = TransformerModule(config).to(config.device)

  @property
  def model(self) -> torch.nn.Module:
    return self._model

  @property
  def window_size(self) -> int:
    return self.config.window_size


class TransformerModule(nn.Module):
  def __init__(self, config: TransformerConfig):
    super(TransformerModule, self).__init__()  # type: ignore
    self.embedding = nn.Linear(config.input_size, config.hidden_size)
    self.transformer = nn.Transformer(
      d_model=config.hidden_size,
      nhead=config.num_heads,
      num_encoder_layers=config.num_layers,
      num_decoder_layers=config.num_layers,
      dim_feedforward=config.dim_feedforward,
      dropout=config.dropout,
      batch_first=True,
    )
    self.fc_out = nn.Linear(config.hidden_size, 1)  # Output layer with one unit for binary classification

  def forward(self, src: torch.Tensor):
    src = self.embedding(src)  # Embed the source sequence
    src = src.permute(1, 0, 2)  # Convert (batch_size, seq_len, embed_dim) to (seq_len, bs, embed_dim)
    transformer_output = self.transformer(src, src)
    # print(transformer_output.shape)
    output = self.fc_out(transformer_output[transformer_output.shape[0] // 2, :, :])
    # print(output.shape)
    output = torch.sigmoid(output)
    return output  # Convert back to (batch_size, seq_len, 1)
