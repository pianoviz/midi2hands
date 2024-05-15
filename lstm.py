import utils as U
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import json


class LSTMModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, device, **kwargs
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
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, out.size(1) // 2, :])
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out


def main():
    # generate a random name
    run_name = U.generate_complex_random_name()

    # instantiate the logge
    logger = U.setup_logger(__name__, "lstm", f"{run_name}.log")

    # Hyperparameters
    h_params = {
        "run_name": run_name,
        "seed": 42,
        "batch_size": 64,
        "num_epochs": 20,
        "window_size": 30,
        "input_size": 2,
        "hidden_size": 16,
        "num_layers": 2,
        "num_classes": 1,
        "device": torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
        "preprocessing_func": U.extract_windows_single,
    }

    # set seed
    torch.manual_seed(h_params["seed"])

    # model

    # get the k-fold split
    k_fold_data = U.k_fold_split(10)
    all_results = {"h_params": h_params}

    for i, fold in tqdm(
        enumerate(k_fold_data), total=len(k_fold_data), desc="Folds", unit="fold"
    ):
        model = LSTMModel(**h_params).to(h_params["device"])

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_paths, val_paths = fold

        train_windows, train_labels = U.extract_windows_from_files(
            train_paths,
            window_size=h_params["window_size"],
            step_size=1,
            preprocess_func=h_params["preprocessing_func"],
        )
        val_windows, val_labels = U.extract_windows_from_files(
            val_paths,
            window_size=h_params["window_size"],
            step_size=1,
            preprocess_func=h_params["preprocessing_func"],
        )

        # create a data loader
        train_dataset = U.MidiDataset(train_windows, train_labels)
        val_dataset = U.MidiDataset(val_windows, val_labels)

        # split dataset in train and validation
        train_loader = DataLoader(
            train_dataset, batch_size=h_params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
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
        all_results[f"fold_{i+1}"] = results
        logger.info(
            "//////////////////////////////////////////////////////////////////"
        )

    with open(f"lstm/{run_name}.json", "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
