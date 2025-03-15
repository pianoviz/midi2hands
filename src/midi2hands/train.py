import os
import tempfile
from typing import Any

import mlflow
import onnx
import torch
from joblib.numpy_pickle import Path
from midiutils import MidiPreprocessor
from mlflow.models.signature import infer_signature
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import midi2hands.utils as utils
from midi2hands.config import Device, LSTMConfig, TrainingConfig
from midi2hands.models.generative import GenerativeHandFormer
from midi2hands.models.interface import HandFormer
from midi2hands.models.torch.lstm import LSTMModel
from midi2hands.models.torch.torch_model import TorchModel


def main(handformer: HandFormer, model: TorchModel, train_config: TrainingConfig):
  run_name = utils.generate_complex_random_name()
  run_path = Path(os.getcwd()) / "runs" / Path(run_name)
  if not run_path.exists():
    run_path.mkdir(parents=True)
  print(f"Run name: {run_name}")

  config = model.config
  torch.manual_seed(train_config.seed)

  # Start MLflow run
  with mlflow.start_run(run_name=run_name) as run:
    # Log parameters
    mlflow.log_param("batch_size", train_config.batch_size)
    mlflow.log_param("num_epochs", train_config.num_epochs)
    mlflow.log_param("patience", train_config.patience)
    mlflow.log_param("device", train_config.device.value)
    mlflow.log_param("use_kfold", train_config.use_kfold)
    mlflow.log_param("use_early_stopping", train_config.use_early_stopping)
    mlflow.log_param("window_size", config.window_size)
    mlflow.log_param("seed", train_config.seed)

    k_fold_data = utils.k_fold_split(train_config.n_folds)

    for i, paths in enumerate(tqdm(k_fold_data, total=len(k_fold_data), unit="fold")):
      # Data preparation
      train_paths, val_paths = paths
      train_paths = train_paths[:1]
      val_paths = val_paths[:1]
      train_windows, train_labels = handformer.extract_windows_from_files(train_paths, window_size=config.window_size)
      val_windows, val_labels = handformer.extract_windows_from_files(
        paths=val_paths,
        window_size=config.window_size,
      )
      train_dataset = utils.MidiDataset(train_windows, train_labels)
      val_dataset = utils.MidiDataset(val_windows, val_labels)
      train_loader: DataLoader[Any] = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
      val_loader: DataLoader[Any] = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)

      criterion: nn.Module = nn.BCELoss()
      optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)

      # Log model parameters
      mlflow.log_param(f"fold_{i}_learning_rate", 0.001)

      utils.train_loop(
        model=model.model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=train_config,
        mlflow_run=run,
      )

      if train_config.inference_eval:
        # used to assess real world performance, takes a while to compute.
        group_accuracies: list[float] = []
        y_true: list[int] = []
        y_pred: list[int] = []
        model.model.eval()
        with torch.no_grad():
          for val_path in tqdm(train_paths):
            _, y_t, y_p = handformer.inference(
              events=MidiPreprocessor().get_midi_events(Path(val_path), max_note_length=100),
              window_size=config.window_size,
              device=config.device.value,
            )
            y_true.extend(y_t)
            y_pred.extend(y_p)
            acc = utils.accuracy(y_t, y_p)
            group_accuracies.append(acc)
            safe_path_key = f"fold_{i}_path_{str(val_path).replace('/', '_').replace('.', '_')}_accuracy"
            mlflow.log_metric(safe_path_key, acc)

        group_accuracy = sum(group_accuracies) / len(group_accuracies)
        inference_accuracy = utils.accuracy(y_true, y_pred)
        # Log inference metrics
        mlflow.log_metric(f"fold_{i}_group_accuracy", group_accuracy)
        mlflow.log_metric(f"fold_{i}_inference_accuracy", inference_accuracy)

      # Save and log model artifacts
      # torch.save(model.model.state_dict(), run_path / "model.pth")
      print(model.model)
      input_example = next(iter(train_loader))[0][:1]
      model.model.eval()
      output_example = model.model(input_example.to(config.device.value))
      signature = infer_signature(input_example.cpu().detach().numpy(), output_example.cpu().detach().numpy())
      mlflow.pytorch.log_model(model.model, f"fold_{i}_pytorch_model", signature=signature)

      with tempfile.NamedTemporaryFile() as f:
        model.to_onnx(f.name)
        onnx_model = onnx.load(f.name)
        mlflow.onnx.log_model(onnx_model, f"fold_{i}_onnx_model", signature=signature)

      if not train_config.use_kfold:
        break


if __name__ == "__main__":
  config = LSTMConfig(device=Device.MPS)
  model = LSTMModel(config)
  handformer = GenerativeHandFormer(model)
  train_config = TrainingConfig(
    batch_size=64, num_epochs=0, patience=3, device=Device.MPS, use_kfold=False, use_early_stopping=False, inference_eval=False
  )

  main(handformer=handformer, model=model, train_config=train_config)
