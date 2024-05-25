import logging
import torch
import random
import numpy as np
from mido.midifiles.midifiles import MidiFile
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, List, Tuple, Any
import shutil
from joblib import Memory

torch.manual_seed(0)


memory = Memory(location="cache", verbose=0)


class NoteEvent:
    def __init__(
        self,
        note: int,
        velocity: int,
        start: int,
        end: int | None = None,
        hand: str | None = None,
    ):
        self.note = note
        self.velocity = velocity
        self.start = start
        self.end = end
        self.hand = hand

    def set_end(self, end):
        self.end = end


class MidiDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


class MidiEventProcessor:
    """
    The purpose of this class is to extract note events from a midi file.
    The extract_note_events method returns a list of NoteEvent objects.
    """

    def __init__(self):
        self.note_events: list[NoteEvent] = []

    def _create_note_event(self, active_notes, midi_message, timestamp, hand: str):
        note_event = NoteEvent(
            midi_message.note, midi_message.velocity, timestamp, hand=hand
        )
        active_notes[midi_message.note] = note_event

    def _process_note_off_event(
        self, note_events, active_notes, midi_message, timestamp
    ):
        note_event = active_notes.get(midi_message.note)
        if note_event and note_event.end is None:
            note_event.set_end(timestamp)
            note_events.append(note_event)
            active_notes[midi_message.note] = None

    def _process_midi_track(self, note_events: list, midi_track, hand: str):
        cumulative_time = 0
        active_notes = {}
        for _, midi_message in enumerate(midi_track):
            cumulative_time += midi_message.time
            if midi_message.type == "note_on":
                self._create_note_event(
                    active_notes, midi_message, cumulative_time, hand
                )
            elif midi_message.type == "note_off":
                self._process_note_off_event(
                    note_events, active_notes, midi_message, cumulative_time
                )

    def _extract_and_process_midi_tracks(self, midi_file_path) -> list[NoteEvent]:
        note_events = []
        midi_file = MidiFile(midi_file_path)
        for i, midi_track in enumerate(midi_file.tracks):
            hand = "right" if i == 1 else "left"
            self._process_midi_track(note_events, midi_track, hand)
        return sorted(note_events, key=lambda x: x.start)

    def extract_note_events(self, midi_file_path: Path) -> list[NoteEvent]:
        return self._extract_and_process_midi_tracks(midi_file_path)


def note_events_to_json(events, output_file_path: Path):
    import json

    json_events = []
    for event in events:
        json_events.append(
            {
                "note": event.note,
                "velocity": event.velocity,
                "start": event.start,
                "end": event.end,
                "hand": event.hand,
            }
        )
    with open(output_file_path, "w") as f:
        json.dump(json_events, f)


def log_parameters(params: dict, logger):
    for key, value in params.items():
        logger.info(f"{key}: {value}")


def get_device() -> str:
    return str(
        torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    )


def random_split(paths, ratios):
    random.seed(0)
    random.shuffle(paths)
    total = len(paths)
    ratios = [int(r * total) for r in ratios]
    return paths[: ratios[0]], paths[ratios[0] :]


def split_data():
    paths = list(Path("data").rglob("*.mid"))
    print(f"Found {len(paths)} midi files")

    # split paths into train and validation
    train_paths, test_paths = random_split(paths, [0.9, 0.1])
    train_paths = list(train_paths)

    # creae directories
    Path("data/train").mkdir(parents=True, exist_ok=True)
    Path("data/test").mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(train_paths):
        new_path = Path(f"data/train/{i:03}-{path.stem}.mid")
        # copy the file to the new path
        shutil.copy(path, new_path)
    for i, path in enumerate(test_paths):
        new_path = Path(f"data/test/{i:03}-{path.stem}.mid")
        # copy the file to the new path
        shutil.copy(path, new_path)


def pad_events(events: list[NoteEvent], window_size: int) -> list[NoteEvent]:
    # pad the events with None values
    m = window_size // 2
    for _ in range(m):
        dummy_note = NoteEvent(note=-1, velocity=-1, start=-1, hand=None)
        dummy_note.set_end(-1)
        events.insert(0, dummy_note)
        events.append(dummy_note)
    return events


def convert_hand_to_number(hand: str | None):
    return 0 if hand == "left" else (1 if hand == "right" else -1)


def preprocess_window_discriminative(note_events: list[NoteEvent]):
    """Convert the list of notes to a numpy array, also normalize the start times"""

    def convert(n: NoteEvent):
        return (n.note, n.start, n.end)

    window = np.array([convert(n) for n in note_events], dtype=np.float32)
    non_pad_indices = np.where(window[:, 0] != -1)[0]
    window[non_pad_indices, 1:3] = window[non_pad_indices, 1:3] / np.max(
        window[non_pad_indices, 2]
    )
    window[non_pad_indices, 0] = (window[non_pad_indices, 0] - 21) / 88
    return window


def preprocess_window_generative(note_events: list[NoteEvent]):
    """Convert the list of notes to a numpy array, also normalize the start times"""

    def convert(n: NoteEvent):
        return (n.note, n.start, n.end, convert_hand_to_number(n.hand))

    window = np.array([convert(n) for n in note_events], dtype=np.float32)
    non_pad_indices = np.where(window[:, 0] != -1)[0]
    window[non_pad_indices, 1:3] = window[non_pad_indices, 1:3] / np.max(
        window[non_pad_indices, 2]
    )
    window[non_pad_indices, 0] = (window[non_pad_indices, 0] - 21) / 88

    return window


ExtractFuncType = Callable[
    [List[NoteEvent], int, int],  # input
    Tuple[List[np.ndarray], List[np.ndarray]],  # output
]


def extract_windows_generative(events, window_size, step_size) -> tuple[list, list]:
    """Extract windows and labels from a list of note events
    Include the label in the
    """
    windows, labels = [], []
    padded_events = pad_events(events=events.copy(), window_size=window_size)
    h = window_size // 2
    for i in range(h, len(events) + h, 1):
        window = padded_events[i - h : i + h]
        preprocessed_window = preprocess_window_generative(window)
        label = convert_hand_to_number(window[h].hand)
        label = np.array([label])
        for i in range(h, window_size):
            preprocessed_window[i, -1] = -1
        windows.append(preprocessed_window)
        labels.append(label)
    return windows, labels


def extract_windows_discriminative(events, window_size, step_size) -> tuple[list, list]:
    """Extract windows and labels from a list of note events
    Include the label in the
    """
    windows, labels = [], []
    padded_events = pad_events(events=events.copy(), window_size=window_size)
    h = window_size // 2
    for i in range(h, len(events) + h, 1):
        window = padded_events[i - h : i + h]
        preprocessed_window = preprocess_window_discriminative(window)
        label = convert_hand_to_number(window[h].hand)
        label = np.array([label])
        windows.append(preprocessed_window)
        labels.append(label)
    return windows, labels


# @memory.cache
def extract_windows_from_files(
    paths, window_size, step_size, preprocess_func: ExtractFuncType
):
    all_windows = []
    all_labels = []
    mp = MidiEventProcessor()
    for path in paths:
        events = mp.extract_note_events(path)
        windows, labels = preprocess_func(events, window_size, step_size)
        all_windows.extend(windows)
        all_labels.extend(labels)
    return np.array(all_windows), np.array(all_labels)


def get_events(path):
    mp = MidiEventProcessor()
    return mp.extract_note_events(path)


def discriminative_inference(mid_path, model, window_size, device):
    print("Using Discriminative inference")
    print(f"Processing {mid_path}")
    events = get_events(mid_path)
    padded_events = pad_events(events.copy(), window_size)
    y_true, y_pred = [], []
    h = window_size // 2
    for i in range(h, len(events) + h, 1):
        window_events = padded_events[i - h : i + h]
        preprocessed_window = preprocess_window_discriminative(window_events)
        label = padded_events[i].hand
        label = convert_hand_to_number(label)
        y_true.append(label)

        tensor_window = torch.tensor(preprocessed_window).unsqueeze(0).to(device)
        output = model(tensor_window)
        output = output.squeeze().cpu().detach().numpy()
        output = 0 if output < 0.5 else 1
        y_pred.append(output)

    return events, y_true, y_pred


def generative_inference(mid_path, model, window_size, device):
    print("Using generative inference")
    print(f"Processing {mid_path}")
    events = get_events(mid_path)
    padded_events = pad_events(events.copy(), window_size)
    y_true, y_pred = [], []
    h = window_size // 2
    for i in range(h, len(events) + h, 1):
        window_events = padded_events[i - h : i + h]
        preprocessed_window = preprocess_window_generative(window_events)
        preprocessed_window[:, -1] = -1  # we don't know the output yet
        prev_out = y_pred[-h:]
        preprocessed_window[: len(prev_out), -1] = prev_out

        label = padded_events[i].hand
        label = convert_hand_to_number(label)
        print(label)
        y_true.append(label)

        tensor_window = (
            torch.tensor(preprocessed_window).float().to(device).unsqueeze(0)
        )
        output = model(tensor_window)
        output = output.squeeze().cpu().detach().numpy()
        output = 0 if output < 0.5 else 1
        y_pred.append(output)
    return events, y_true, y_pred


# def generative_accuracies(paths, model, window_size, device):
#     y_true, y_pred = [], []
#     for path in paths:
#         events, y_true_temp, y_pred_temp = get_generative_inference_results(
#             path, model, window_size, device
#         )
#         y_pred.extend(y_pred_temp)
#         y_true.extend(y_true_temp)

#         # song accuracy
#         acc_temp = np.sum(np.array(y_true_temp) == np.array(y_pred_temp)) / len(
#             y_true_temp
#         )
#         print(f"Song accuracy: {acc_temp}")
#     return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)


def accuracy(y_true: list[int], y_pred: list[int]):
    return np.mean(np.array(y_true) == np.array(y_pred))


def setup_logger(name: str, logdir: str, filename: str) -> logging.Logger:
    path = Path(logdir)
    if not path.exists():
        path.mkdir(parents=True)
    path = path / filename
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)  # add the file handler
    return logger


def generate_complex_random_name():
    adjectives = [
        "nostalgic",
        "wonderful",
        "mystic",
        "quiet",
        "vibrant",
        "eager",
        "frosty",
        "peaceful",
        "serene",
        "ancient",
    ]
    nouns = [
        "morse",
        "turing",
        "neumann",
        "lovelace",
        "hopper",
        "tesla",
        "einstein",
        "bohr",
        "darwin",
        "curie",
    ]
    numbers = range(10, 99)  # two digit numbers

    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    number = random.choice(numbers)
    return f"{adjective}_{noun}_{number}"


def k_fold_split(k, max_files=None):
    paths = list(Path("data").rglob("*.mid"))
    if max_files:
        paths = paths[: max_files + 1]
    n = len(paths)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        val_fold = paths[start:end]
        train_fold = paths[:start] + paths[end:]
        folds.append((train_fold, val_fold))
    return folds


def process_batch(windows, labels, model, criterion, device):
    windows = windows.to(device)
    labels = labels.float().to(device)
    outputs = model(windows)

    loss = criterion(outputs, labels)

    # Simplify handling of batches with single sample
    labels = labels.squeeze()
    outputs = outputs.squeeze()
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)
    if outputs.ndim == 0:
        outputs = outputs.unsqueeze(0)

    y_true = labels.cpu().numpy().astype(int).tolist()
    y_pred = np.where(outputs.cpu().detach().numpy() > 0.5, 1, 0).tolist()

    return loss, y_true, y_pred


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    logger,
    use_early_stopping,
    patience,
    device,
    num_epochs,
    **extra_params,
):
    # Store metrics for all epochs
    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if use_early_stopping:
        num_epochs = 500

    # Function to process batches

    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses, train_accs = [], []
        for windows, labels in train_loader:
            loss, y_t, y_p = process_batch(windows, labels, model, criterion, device)
            train_losses.append(loss.item())
            train_accs.append(accuracy(y_t, y_p))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Logging training metrics
        metrics["train_loss"].append(np.mean(train_losses))
        metrics["train_acc"].append(np.mean(train_accs))

        # Validation phase
        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for windows, labels in val_loader:
                loss, y_t, y_p = process_batch(
                    windows, labels, model, criterion, device
                )
                val_losses.append(loss.item())
                val_accs.append(accuracy(y_t, y_p))

        # Logging validation metrics
        metrics["val_loss"].append(np.mean(val_losses))
        metrics["val_acc"].append(np.mean(val_accs))

        # Output log for each epoch
        logger.info(
            f"Epoch {epoch+1:02}\ttrain_loss: {metrics['train_loss'][-1]:.4f}\t"
            f"train_acc: {metrics['train_acc'][-1]:.4f}\t"
            f"val_loss: {metrics['val_loss'][-1]:.4f}\t"
            f"val_acc: {metrics['val_acc'][-1]:.4f}"
        )

        # Early stopping
        if use_early_stopping:
            if metrics["val_loss"][-1] < best_val_loss:
                best_val_loss = metrics["val_loss"][-1]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        f"Early stopping after {epoch+1} epochs without improvement"
                    )
                    break
    return metrics


if __name__ == "__main__":
    split_data()
