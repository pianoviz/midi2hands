from joblib.numpy_pickle import Path
import utils as U
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import os
import argparse

from models.lstm import LSTMModel
from models.handformer import HandformerModel


def hand_former_model(h_params):
    return HandformerModel(**h_params).to(h_params["device"])


def lstm_model(h_params):
    return LSTMModel(**h_params).to(h_params["device"])


def read_h_params(json_path):
    with open(json_path, "r") as f:
        h_params = json.load(f)
    return h_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    args = parser.parse_args()
    run_name = U.generate_complex_random_name()
    run_path = Path(os.getcwd()) / "results" / Path(run_name)
    if not run_path.exists():
        run_path.mkdir(parents=True)

    print(f"Run name: {run_name}")
    logger = U.setup_logger(__name__, str(run_path), "log")

    device = U.get_device()

    h_params = read_h_params(args.config_file)

    # set extra params
    h_params["run_name"] = run_name
    h_params["device"] = str(device)

    logger.info("Running with fixed parameters:")
    U.log_parameters(h_params, logger)
    logger.info("===========================================================\n\n")

    torch.manual_seed(h_params["seed"])

    k_fold_data = U.k_fold_split(h_params["n_folds"])

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

        model = eval(h_params["model_func"])(h_params)
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
        if not h_params["use_kfold"]:
            break

    with open(run_path / "results.json", "w") as f:
        print(all_results)
        json.dump(all_results, f, indent=4)

    # save model
    if model:
        torch.save(model.state_dict(), run_path / "model.pth")


if __name__ == "__main__":
    main()
