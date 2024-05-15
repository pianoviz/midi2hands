import torch
from torch import nn
import torch.optim as optim

import utils as U
from tqdm import tqdm
from torch.utils.data import DataLoader
import json

import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(
            embed_dim, 1
        )  # Output layer with one unit for binary classification

    def forward(self, src):
        src = self.embedding(src)  # Embed the source sequence
        src = self.positional_encoding(src)
        src = src.permute(
            1, 0, 2
        )  # Convert (batch_size, seq_len, embed_dim) to (seq_len, batch_size, embed_dim)
        transformer_output = self.transformer(src, src)
        output = self.fc_out(transformer_output)
        return output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, 1)


def main():
    # generate a random name
    run_name = U.generate_complex_random_name()

    # instantiate the logge
    logger = U.setup_logger(__name__, "transformer", f"{run_name}.log")

    # Hyperparameters
    h_params = {
        "run_name": run_name,
        "seed": 42,
        "batch_size": 64,
        "num_epochs": 20,
        "window_size": 30,
        "input_size": 2,
        "hidden_size": 16,
        "num_layers": 2,
        "num_classes": 1,
        "device": torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
        "preprocessing_func": U.extract_windows_seq_2_seq,
    }

    # set seed
    torch.manual_seed(h_params["seed"])

    # model

    # get the k-fold split
    k_fold_data = U.k_fold_split(10)
    all_results = {"h_params": h_params}

    # Hyperparameters
    input_dim = 2  # For MIDI pitch and relative start time
    embed_dim = 64
    num_heads = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 256
    dropout = 0.1

    # Initialize the model, loss function, and optimizer

    for i, fold in tqdm(
        enumerate(k_fold_data), total=len(k_fold_data), desc="Folds", unit="fold"
    ):
        train_paths, val_paths = fold
        model = TransformerModel(
            input_dim,
            embed_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
        ).to(h_params["device"])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_windows, train_labels = U.extract_windows_from_files(
            train_paths,
            window_size=h_params["window_size"],
            step_size=1,
            preprocess_func=h_params["preprocessing_func"],
        )
        val_windows, val_labels = U.extract_windows_from_files(
            val_paths,
            window_size=h_params["window_size"],
            step_size=1,
            preprocess_func=h_params["preprocessing_func"],
        )

        # create a data loader
        train_dataset = U.MidiDataset(train_windows, train_labels)
        val_dataset = U.MidiDataset(val_windows, val_labels)

        # split dataset in train and validation
        train_loader = DataLoader(
            train_dataset, batch_size=h_params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=h_params["batch_size"], shuffle=False
        )

        # train the model
        results = U.train_loop(
            model,
            train_loader,
            val_loader,
            h_params["num_epochs"],
            optimizer,
            criterion,
            h_params["device"],
            logger,
        )
        all_results[f"fold_{i+1}"] = results
        logger.info(
            "//////////////////////////////////////////////////////////////////"
        )

    with open(f"transformer/{run_name}.json", "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
