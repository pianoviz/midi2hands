from abc import ABC, abstractmethod

import torch

from midi2hands.config import BaseConfig, Device, LSTMConfig, TrainingConfig, TransformerConfig
from midi2hands.models.generative import GenerativeHandFormer
from midi2hands.models.interface import HandFormer
from midi2hands.models.torch.lstm import LSTMModel
from midi2hands.models.torch.transformer import TransformerModel


class ModelSpec(ABC):
  @property
  @abstractmethod
  def config(self) -> BaseConfig: ...

  @property
  @abstractmethod
  def train_config(self) -> TrainingConfig: ...

  @property
  @abstractmethod
  def handformer(self) -> HandFormer: ...

  @property
  @abstractmethod
  def model(self) -> torch.nn.Module: ...


class GenerativeTransformer(ModelSpec):
  def __init__(self):
    self._config = TransformerConfig()
    self._model = GenerativeHandFormer(model=TransformerModel(self._config))

  @property
  def config(self) -> BaseConfig:
    return self._config

  @property
  def train_config(self) -> TrainingConfig:
    return TrainingConfig(batch_size=16, num_epochs=2, device=Device.MPS)

  @property
  def model(self) -> torch.nn.Module:
    return self._model.model.model

  @property
  def handformer(self) -> HandFormer:
    return self._model


class GenerativeLSTM(ModelSpec):
  def __init__(self):
    self._config = LSTMConfig(device=Device.MPS)
    self._model = GenerativeHandFormer(model=LSTMModel(self._config))

  @property
  def config(self) -> BaseConfig:
    return self._config

  @property
  def train_config(self) -> TrainingConfig:
    return TrainingConfig(batch_size=64, num_epochs=20, patience=3, device=Device.MPS, use_kfold=False)

  @property
  def model(self) -> torch.nn.Module:
    return self._model.model.model

  @property
  def handformer(self) -> HandFormer:
    return self._model


if __name__ == "__main__":
  model_spec = GenerativeTransformer()
