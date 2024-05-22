import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, device, **extra_params
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            self.device
        )
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            self.device
        )
        # print(x.shape, h0.shape, c0.shape)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, out.size(1) // 2, :])
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out
