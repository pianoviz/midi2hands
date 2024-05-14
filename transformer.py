import torch
from torch import nn

import math


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, window_size, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(window_size, hidden_size)
        position = torch.arange(0, window_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape is now [1, window_size, hidden_size]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape is assumed to be [batch_size, window_size, hidden_size]
        x = (
            x + self.pe[: x.size(1), :]
        )  # Corrected to add along the window_size dimension
        return self.dropout(x)


class HandFormer(nn.Module):
    def __init__(self, num_features, num_layers, hidden_size, num_heads, window_size):
        super(HandFormer, self).__init__()
        self.window_size = window_size
        self.embedding = nn.Linear(num_features, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, window_size)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                batch_first=True, d_model=hidden_size, nhead=num_heads
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_size, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        # embed the notes
        # x.shape is (batch_size, window_size, 3)
        # print(f'x.shape: {x.shape}')
        x = self.embedding(x)
        # x = self.pos_encoder(x) # Add positional encoding
        # x.shape is (batch_size, window_size, hidden_size)
        # print(f'x.shape: {x.shape}')
        x = self.encoder(x)
        # x = torch.sum(x, dim=1) # Example aggregation (summing all the output states)
        # x.shape is (batch_size, window_size, hidden_size)
        x = self.fc(x[:, 14, :])
        x = self.fc2(x)
        # x.shape is (batch_size, window_size, 1)
        # print(f'x.shape: {x.shape}')
        return x
