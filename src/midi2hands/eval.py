import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import midi2hands.utils as utils

# def get_model(model_dir, h_params, device):
#   """Try to read the model from the state_dict file"""

#   model = eval(h_params["model_func"])(h_params)
#   model.load_state_dict(
#     torch.load(model_dir / "model.pth", map_location=device)
#   )
#   model.eval()
#   return model


def get_test_files():
  return list(Path("data/test").glob("*.mid"))


def accuracy_loss_plot(save_dir: Path, filename: str, results: dict[str, list[float]], fold: int = 0):
  val_acc = results[f"fold_{fold}"]["val_acc"]  # type: ignore
  val_loss = results[f"fold_{fold}"]["val_loss"]  # type: ignore
  train_acc = results[f"fold_{fold}"]["train_acc"]  # type: ignore
  train_loss = results[f"fold_{fold}"]["train_loss"]  # type: ignore
  fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))  # type: ignore
  ax[0].plot(val_acc, label="Validation accuracy")
  ax[0].plot(train_acc, label="Train accuracy")
  ax[0].set_title("Accuracy")
  ax[0].legend()

  ax[1].plot(val_loss, label="Validation loss")
  ax[1].plot(train_loss, label="Train loss")
  ax[1].set_title("Loss")
  ax[1].legend()

  plt.show()  # type: ignore
  # save fig
  fig.savefig(save_dir / filename)  # type: ignore


"""
def to_onnx(model_dir, filename, window_size, input_size, **h_params):
  # convert to onnx
  model = get_model(model_dir, h_params, "cpu")
  model.eval()
  dummy_input = torch.randn(1, window_size, input_size)
  torch.onnx.export(model, (dummy_input,), model_dir / filename)
"""


# TODO: fix the inference accuracy
# def get_test_acc(
#   handformer: HandFormer,
# ):
#   test_paths = get_test_files()
#   y_true, y_pred = [], []
#   for mid_path in test_paths:
#     events_original = MidiPreprocessor().get_midi_events(
#       mid_path, max_note_length=100
#     )
#     events, y_true, y_pred = handformer.inference(
#       events_original,
#       model,
#       h_params["window_size"],
#     )
#     print(f"accuracy: {utils.accuracy(y_true, y_pred)}")
#   print(classification_report(y_true, y_pred))


def get_k_fold_results(model_dir: Path):
  # get the results of the k-fold

  with open(model_dir / "results.json", "r") as f:
    results = json.load(f)
  # create a pandas dataframe with the results
  data: dict[str, list[float]] = {
    "fold": [],
    "train_acc": [],
    "train_loss": [],
    "val_acc": [],
    "val_loss": [],
    "generative_acc_mean": [],
    "generative_acc_std": [],
  }
  fold_keys = [key for key in results.keys() if "fold" in key]
  for fold in fold_keys:
    data["fold"].append(fold)
    data["train_acc"].append(np.max(results[fold]["train_acc"]))
    data["train_loss"].append(np.min(results[fold]["train_loss"]))
    data["val_acc"].append(np.max(results[fold]["val_acc"]))
    data["val_loss"].append(np.min(results[fold]["val_loss"]))
    data["generative_acc_mean"].append(float(np.mean([results[fold]["generative_accuracy"]])))
    data["generative_acc_std"].append(float(np.std(results[fold]["generative_accuracy"])))

  df = pd.DataFrame(data)

  # print the results
  print(df)
  # print a latex table of the results, use 4 decimal places
  print(df.to_latex(float_format="%.4f"))  # type: ignore


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", type=str, required=True)
  args = parser.parse_args()

  # results and h_params
  model_dir = Path(args.model_dir)
  with open(model_dir / "results.json", "r") as f:
    results = json.load(f)
  h_params = results["h_params"]

  video = True
  to_onnex = False
  inference_acc = False
  train_stats = False
  k_fold_results = False

  device = utils.get_device()
  h_params["device"] = str(device)

  # model = get_mdel(model_dir, h_params, device).to(device)
  test_paths = get_test_files()

  # if video:
  #   generate_videos(
  #     mid_path=test_paths[1],
  #     model_dir=model_dir,
  #     model=model,
  #     **h_params,
  #   )
  # if inference_acc:
  #   get_test_acc(model, h_params)
  # if to_onnex:
  #   to_onnx(model_dir, "model.onnx", **h_params)
  # if train_stats:
  #   accuracy_loss_plot(model_dir, "training.png", results, fold=0)
  # if k_fold_results:
  #   get_k_fold_results(model_dir, results)
