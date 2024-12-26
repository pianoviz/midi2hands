import json
from pathlib import Path
from typing import Any, Dict, Literal


class Config:
  model: Literal["transformer", "lstm"]
  device: Literal["cpu", "cuda", "mps"]

  def __init__(
    self,
    model: Literal["transformer", "lstm"],
    seed: int = 42,
    batch_size: int = 64,
    num_epochs: int = 2,
    window_size: int = 30,
    input_size: int = 3,
    hidden_size: int = 32,
    dropout: float = 0.1,
    num_layers: int = 3,
    num_classes: int = 1,
    n_folds: int = 10,
    use_early_stopping: bool = True,
    patience: int = 5,
    model_func: str = "U.lstm_model",
    use_kfold: bool = True,
    run_name: str | None = None,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
  ) -> None:
    self.model = model
    self.seed = seed
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.window_size = window_size
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.dropout = dropout
    self.num_layers = num_layers
    self.num_classes = num_classes
    self.n_folds = n_folds
    self.use_early_stopping = use_early_stopping
    self.patience = patience
    self.model_func = model_func
    self.use_kfold = use_kfold
    self.run_name = run_name
    self.device = device

  @classmethod
  def from_json(cls, json_path: Path) -> "Config":
    with open(json_path, "r") as f:
      data: Dict[str, Any] = json.load(f)
    # data:  = json.loads(json_data)
    return cls(
      model=data.get("model", "generativeTransformer"),
      seed=data.get("seed", 42),
      batch_size=data.get("batch_size", 64),
      num_epochs=data.get("num_epochs", 2),
      window_size=data.get("window_size", 30),
      input_size=data.get("input_size", 3),
      hidden_size=data.get("hidden_size", 32),
      dropout=data.get("dropout", 0.1),
      num_layers=data.get("num_layers", 3),
      num_classes=data.get("num_classes", 1),
      n_folds=data.get("n_folds", 10),
      use_early_stopping=data.get("use_early_stopping", True),
      patience=data.get("patience", 5),
      use_kfold=data.get("use_kfold", True),
    )

  def __repr__(self) -> str:
    return f"""Config(\
        seed={self.seed}
        batch_size={self.batch_size}
        num_epochs={self.num_epochs}
        window_size={self.window_size}
        input_size={self.input_size}
        hidden_size={self.hidden_size}
        dropout={self.dropout}
        num_layers={self.num_layers}
        num_classes={self.num_classes}
        n_folds={self.n_folds}
        use_early_stopping={self.use_early_stopping}
        patience={self.patience}
        use_kfold={self.use_kfold}
        )"""
