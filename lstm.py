import utils as U
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import itertools


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
    run_name = U.generate_complex_random_name()
    logger = U.setup_logger(__name__, "lstm", f"{run_name}.log")

    h_params = {
        "run_name": run_name,
        "max_files": 20,
        "seed": 42,
        "batch_size": 64,
        "num_epochs": 20,
        "window_size": 30,
        "input_size": 3,
        "hidden_size": 16,
        "num_layers": 2,
        "num_classes": 1,
        "n_folds": 10,
        "device": torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
        "preprocessing_func": U.extract_windows_single,
        "use_kfold": False,
    }
    logger.info(f"Running with parameters:")
    U.log_parameters(h_params, logger)
    logger.info("========================================\n")

    torch.manual_seed(h_params["seed"])

    param_grid = {
        "hidden_size": [16, 32, 64],
        "num_layers": [1, 2, 3],
        "batch_size": [32, 64, 128],
    }
    param_grid = {}

    all_results: dict[str, float | dict | list | None] = {"h_params": h_params}

    k_fold_data = U.k_fold_split(h_params["n_folds"], max_files=h_params["max_files"])

    if not h_params["use_kfold"]:
        k_fold_data = [k_fold_data[0]]

    best_val_loss = float("inf")
    best_h_params = None

    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        logger.info(f"\nTraining with params:")
        U.log_parameters(param_dict, logger)
        h_params.update(param_dict)

        fold_results = []

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

            train_dataset = U.MidiDataset(train_windows, train_labels)
            val_dataset = U.MidiDataset(val_windows, val_labels)

            train_loader = DataLoader(
                train_dataset, batch_size=h_params["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=h_params["batch_size"], shuffle=False
            )

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
            fold_results.append(results["val_loss"])

        avg_val_loss = sum(fold_results) / len(fold_results)
        if avg_val_loss < best_val_loss:
            logger.info(f"New best validation loss: {avg_val_loss}")
            best_val_loss = avg_val_loss
            best_h_params = param_dict

        all_results[f"params_{params}"] = fold_results

        logger.info(f"Validation loss for params {params}: {avg_val_loss}")

    all_results["best_h_params"] = best_h_params
    all_results["best_val_loss"] = best_val_loss

    with open(f"lstm/{run_name}.json", "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
