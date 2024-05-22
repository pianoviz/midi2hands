from joblib.numpy_pickle import Path
import utils as U
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import os

from models.lstm import LSTMModel


def main():
    run_name = U.generate_complex_random_name()
    run_path = Path(os.getcwd()) / "lstm" / Path(run_name)
    if not run_path.exists():
        run_path.mkdir(parents=True)

    print(f"Run name: {run_name}")
    logger = U.setup_logger(__name__, str(run_path), f"log")

    h_params = {
        "run_name": run_name,
        "max_files": None,
        "seed": 42,
        "batch_size": 64,
        "num_epochs": 2,
        "window_size": 30,
        "input_size": 3,
        "hidden_size": 16,
        "num_layers": 2,
        "num_classes": 1,
        "n_folds": 10,
        "use_early_stopping": True,
        "patience": 3,
        "device": str(
            torch.device(
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        ),
        "preprocessing_func": "U.extract_windows_single",
        "use_kfold": False,
    }
    logger.info("Running with fixed parameters:")
    U.log_parameters(h_params, logger)
    logger.info("===========================================================\n\n")

    torch.manual_seed(h_params["seed"])

    k_fold_data = U.k_fold_split(h_params["n_folds"], max_files=h_params["max_files"])

    best_val_loss = float("inf")

    model = None
    all_results = {}
    for i, paths in enumerate(
        tqdm(
            k_fold_data,
            total=len(k_fold_data),
            unit="fold",
        )
    ):
        train_paths, val_paths = paths

        model = LSTMModel(**h_params).to(h_params["device"])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_windows, train_labels = U.extract_windows_from_files(
            train_paths,
            window_size=h_params["window_size"],
            step_size=1,
            preprocess_func=eval(h_params["preprocessing_func"]),
        )
        val_windows, val_labels = U.extract_windows_from_files(
            val_paths,
            window_size=h_params["window_size"],
            step_size=1,
            preprocess_func=eval(h_params["preprocessing_func"]),
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
            optimizer,
            criterion,
            logger,
            **h_params,
        )

        all_results[f"fold_{i}"] = results
        # running generative_accuracy and appending to results
        generative_accuracy = U.generative_accuracy(
            model, val_paths, h_params["window_size"], h_params["device"]
        )
        results["generative_accuracy"] = generative_accuracy

        if not h_params["use_kfold"]:
            break

    with open(run_path / "results.json", "w") as f:
        print(all_results)
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
