import argparse
import json
import os
from typing import Any

import torch
from joblib.numpy_pickle import Path
from midiutils.midi_preprocessor import MidiPreprocessor
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Literal

import midi2hands.utils as utils
from midi2hands.config import Config
from midi2hands.models.generative import GenerativeHandFormer
from midi2hands.models.torch.lstm import LSTMModel
from midi2hands.models.torch.transformer import TransformerModel


def get_model(model_type: Literal["transformer", "lstm"], config: Config):
  if model_type == "transformer":
    return TransformerModel(config=config)
  if model_type == "lstm":
    return LSTMModel(config=config)


def main():
  # args
  parser = argparse.ArgumentParser()
  parser.add_argument("--config_file", required=True)
  args = parser.parse_args()

  # paths and run name
  run_name = utils.generate_complex_random_name()
  run_path = Path(os.getcwd()) / "weights" / Path(run_name)
  if not run_path.exists():
    run_path.mkdir(parents=True)
  print(f"Run name: {run_name}")
  logger = utils.setup_logger(__name__, str(run_path), "log")

  # config
  config = Config.from_json(args.config_file)
  config.run_name = run_name
  config.device = utils.get_device()

  print(f"running on device: {config.device}")

  logger.info("Running with fixed parameters:")
  # utils.log_parameters(h_params, logger)
  print(config)
  logger.info(config)
  logger.info("===========================================================\n\n")

  torch.manual_seed(config.seed)  # type: ignore

  k_fold_data = utils.k_fold_split(config.n_folds)

  all_results = {}
  all_results["h_params"] = str(Config)
  for i, paths in enumerate(
    tqdm(
      k_fold_data,
      total=len(k_fold_data),
      unit="fold",
    )
  ):
    train_paths, val_paths = paths

    model = get_model(model_type=config.model, config=config)
    handformer = GenerativeHandFormer(model=model)
    criterion: nn.Module = nn.BCELoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)

    train_windows, train_labels = handformer.extract_windows_from_files(train_paths, window_size=config.window_size)
    val_windows, val_labels = handformer.extract_windows_from_files(
      paths=val_paths,
      window_size=config.window_size,
    )

    train_dataset = utils.MidiDataset(train_windows, train_labels)
    val_dataset = utils.MidiDataset(val_windows, val_labels)

    train_loader: DataLoader[Any] = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader: DataLoader[Any] = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    results = utils.train_loop(
      model=model.model,
      train_loader=train_loader,
      val_loader=val_loader,
      optimizer=optimizer,
      criterion=criterion,
      logger=logger,
      config=config,
    )
    all_results[f"fold_{i}"] = results
    group_accuracies: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    model.model.eval()
    with torch.no_grad():
      for val_path in train_paths:
        _, y_t, y_p = handformer.inference(
          events=MidiPreprocessor().get_midi_events(Path(val_path), max_note_length=100),
          window_size=config.window_size,
          device=config.device,
        )
        y_true.extend(y_t)
        y_pred.extend(y_p)

        acc = utils.accuracy(y_t, y_p)
        group_accuracies.append(acc)

    group_accuracy = sum(group_accuracies) / len(group_accuracies)
    inference_accuracy = utils.accuracy(y_true, y_pred)
    all_results[f"fold_{i}"]["group_accuracy"] = group_accuracy
    all_results[f"fold_{i}"]["inference_accuracy"] = inference_accuracy

    logger.info(f"Inference group accuracy mean: {group_accuracy}")
    logger.info(f"Inference mean: {inference_accuracy}")

    # save model
    torch.save(model.model.state_dict(), run_path / "model.pth")  # type: ignore
    if not config.use_kfold:
      break

  with open(run_path / "results.json", "w") as f:
    json.dump(all_results, f, indent=4)


if __name__ == "__main__":
  main()
