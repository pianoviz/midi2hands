from models.lstm import LSTMModel
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
import utils as U
import numpy as np


def get_model(model_dir, h_params, device):
    """Try to read the model from the state_dict file"""

    # model = LSTMModel(
    #     input_size=h_params["input_size"],
    #     hidden_size=h_params["hidden_size"],
    #     num_layers=h_params["num_layers"],
    #     num_classes=h_params["num_classes"],
    #     device=device,
    # )
    model = LSTMModel(
        input_size=3,
        hidden_size=64,
        num_layers=3,
        num_classes=1,
        device=device,
    )
    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=device))
    model.eval()
    return model


def get_test_files():
    return Path("data/test").glob("*.mid")


def accuracy_loss_plot(
    save_dir: Path, filename: str, val_acc, val_loss, train_acc, train_loss
):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    ax[0].plot(val_acc, label="Validation accuracy")
    ax[0].plot(train_acc, label="Train accuracy")
    ax[0].set_title("Accuracy")
    ax[0].legend()

    ax[1].plot(val_loss, label="Validation loss")
    ax[1].plot(train_loss, label="Train loss")
    ax[1].set_title("Loss")
    ax[1].legend()

    plt.show()
    # save fig
    fig.savefig(save_dir / filename)


def evaluate_model(model, test_loader, device):
    criterion = nn.BCELoss()
    model.to(device)

    model.eval()
    test_losses, test_accs = [], []
    y_true, y_pred = [], []
    with torch.no_grad():
        for windows, labels in test_loader:
            loss, y_t, y_p = U.process_batch(windows, labels, model, criterion, device)
            y_true.extend(y_t)
            y_pred.extend(y_p)
    return y_true, y_pred


def grid_search_scatter(grid_params):
    """
    x-axis: accuracy
    y-axis: loss
    label: hyper parameters
    "grid_params": [
        {
            "params": {
                "hidden_size": 32,
                "num_layers": 1,
                "batch_size": 32,
                "window_size": 16
            },
            "results": {
                "val_loss": [
                    0.20729194543327614,
                    ...
                    0.19080950682529232
                ],
                "val_acc": [
                    0.9199026869936201,
                    ...
                    0.9284903518728718
                ]
            }
        },
    """
    fig, ax = plt.subplots()
    for params in grid_params:
        val_acc = params["results"]["val_acc"]
        val_loss = params["results"]["val_loss"]
        ax.scatter(val_acc, val_loss, label=params["params"])
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()


def to_onnx(model_dir, filename, window_size, input_size, **h_params):
    # convert to onnx
    model = get_model(model_dir, h_params, "cpu")
    model.eval()
    dummy_input = torch.randn(1, window_size, input_size)
    torch.onnx.export(model, dummy_input, model_dir / filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    # read the results.json file

    with open(model_dir / "results.json", "r") as f:
        results = json.load(f)

    h_params = results["h_params"]
    device = U.get_device()

    test_paths = get_test_files()

    test_windows, train_labels = U.extract_windows_from_files(
        test_paths,
        window_size=h_params["window_size"],
        step_size=1,
        preprocess_func=eval(h_params["preprocessing_func"]),
    )

    test_dataset = U.MidiDataset(test_windows, train_labels)
    test_loader = DataLoader(
        test_dataset, batch_size=h_params["batch_size"], shuffle=False
    )

    model = get_model(model_dir, h_params, device)
    y_true, y_pred = evaluate_model(model, test_loader, device)

    print(classification_report(y_true, y_pred))

    to_onnx(model_dir, "model.onnx", **h_params)

    # analysis of the training process with the results.json file
    # get the best model training results
    grid_params: list[dict[str, dict[str, int]]] = results["grid_params"]
    grid_search_scatter(grid_params)

    # plot the accuracy and loss of the best model
    accuracy_loss_plot(
        model_dir,
        "best_model_training.png",
        results["val_acc"],
        results["val_loss"],
        results["train_acc"],
        results["train_loss"],
    )
