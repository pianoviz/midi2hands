import torch
from torch import nn


class HandformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout,
        **kwargs,
    ):
        super(HandformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(
            hidden_size, 1
        )  # Output layer with one unit for binary classification

    def forward(self, src):
        src = self.embedding(src)  # Embed the source sequence
        src = src.permute(
            1, 0, 2
        )  # Convert (batch_size, seq_len, embed_dim) to (seq_len, batch_size, embed_dim)
        transformer_output = self.transformer(src, src)
        # print(transformer_output.shape)
        output = self.fc_out(transformer_output[transformer_output.shape[0] // 2, :, :])
        # print(output.shape)
        output = torch.sigmoid(output)
        return output  # Convert back to (batch_size, seq_len, 1)
