from joblib.numpy_pickle import Path
import utils as U
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import itertools
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

    param_grid = {
        "hidden_size": [32, 64, 128],
        "num_layers": [1, 3],
        "batch_size": [32, 64, 128],
        "window_size": [16, 32, 64],
    }
    combinations = list(itertools.product(*param_grid.values()))
    print(f"number of combinations: {len(combinations)}")
    # param_grid = {
    #     "window_size": [5, 10],
    # }

    all_results: dict[str, float | dict | list | None] = {"h_params": h_params}
    all_results["grid_params"] = []

    k_fold_data = U.k_fold_split(h_params["n_folds"], max_files=h_params["max_files"])

    if not h_params["use_kfold"]:
        k_fold_data = [k_fold_data[0]]

    best_val_loss = float("inf")

    model = None
    results = {}
    for params in tqdm(
        combinations,
        total=len(combinations),
        desc="Grid search",
        unit="combination",
    ):
        param_dict = dict(zip(param_grid.keys(), params))
        logger.info("Training with params:")
        U.log_parameters(param_dict, logger)
        h_params.update(param_dict)

        fold_results = []

        for i, fold in enumerate(k_fold_data):
            model = LSTMModel(**h_params).to(h_params["device"])

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train_paths, val_paths = fold

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
            fold_results.append(results["val_loss"][-1])  # last epoch val
            # do next fold

        # save results for this set of parameters

        # vi vill spara kombinationen av grid params och basta loss/acc

        if not h_params["use_kfold"]:
            all_results["grid_params"].append(
                {
                    "params": param_dict,
                    "results": {
                        "val_loss": results["val_loss"],
                        "val_acc": results["val_acc"],
                    },
                }
            )

        avg_val_loss = sum(fold_results) / len(fold_results)
        if avg_val_loss < best_val_loss:
            logger.info(f"New best validation loss: {avg_val_loss}")
            best_val_loss = avg_val_loss
            # save model checkpoint
            if model:
                torch.save(model.state_dict(), Path(run_path / "best_model.pth"))
            all_results["val_loss"] = results["val_loss"]
            all_results["val_acc"] = results["val_acc"]
            all_results["train_loss"] = results["train_loss"]
            all_results["train_acc"] = results["train_acc"]
            all_results["h_params"] = h_params
            all_results["best_val_loss"] = best_val_loss

        # all_results[f"params_{params}"] = fold_results

        logger.info(f"Validation loss for params {params}: {avg_val_loss}")

    with open(run_path / "results.json", "w") as f:
        print(all_results)
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
