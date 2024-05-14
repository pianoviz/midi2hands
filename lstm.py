import utils as U
import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out


def main():
    # instantiate the logger
    logger = U.setup_logger("lstm", "lstm.log")

    # Hyperparameters
    h_params = {
        "seed": 42,
        "batch_size": 64,
        "num_epochs": 20,
        "input_size": 3,
        "hidden_size": 16,
        "num_layers": 2,
        "num_classes": 1,
        "device": torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    }

    # set seed
    torch.manual_seed(h_params["seed"])

    # model
    model = LSTMModel(**h_params)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_windows, train_labels = U.extract_windows_from_files(
        train_paths, window_size=60, step_size=1
    )
    val_windows, val_labels = U.extract_windows_from_files(
        val_paths, window_size=60, step_size=1
    )

    # create a data loader
    train_dataset = U.MidiDataset(train_windows, train_labels)
    val_dataset = U.MidiDataset(val_windows, val_labels)

    # split dataset in train and validation
    train_loader = U.DataLoader(
        train_dataset, batch_size=h_params["batch_size"], shuffle=True
    )
    val_loader = U.DataLoader(
        val_dataset, batch_size=h_params["batch_size"], shuffle=False
    )

    # train the model
    results = U.train_loop(
        model,
        train_loader,
        val_loader,
        h_params["num_epochs"],
        optimizer,
        criterion,
        h_params["device"],
        logger,
    )


if __name__ == "__main__":
    main()
