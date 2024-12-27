from abc import ABC, abstractmethod

from midi2hands.config import BaseConfig, TrainingConfig, TransformerConfig
from midi2hands.models.generative import GenerativeHandFormer
from midi2hands.models.interface import HandFormer
from midi2hands.models.torch.torch_model import TorchModel
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
  def model(self) -> TorchModel: ...


class GenerativeTransformer(ModelSpec):
  def __init__(self):
    self.config: TransformerConfig = TransformerConfig()
    self.train_config: TrainingConfig = TrainingConfig()
    _model = GenerativeHandFormer(model=TransformerModel(self.config))
    self.model: TorchModel = _model.model.model
    self.handformer: HandFormer = _model
