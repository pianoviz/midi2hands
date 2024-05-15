import logging
import torch
import random
import numpy as np
from mido.midifiles.midifiles import MidiFile
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, List, Tuple, Any

from joblib import Memory

memory = Memory(location="cache", verbose=0)


class NoteEvent:
    def __init__(self, note: int, velocity: int, start: int, hand: str | None = None):
        self.note = note
        self.velocity = velocity
        self.start = start
        self.end = None
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
            midi_message.note, midi_message.velocity, timestamp, hand
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


def preprocess_window(note_events: list[NoteEvent]):
    """Convert the list of notes to a numpy array, also normalize the start times"""
    window = np.array([(n.note, n.start) for n in note_events], dtype=np.float32)
    # move the pitch values so that they correspond to the the 88 notes on a piano and normalize
    window[:, 0] = (window[:, 0] - 21) / 88
    window[:, 1] = window[:, 1] / window[-1, 1]
    return window


ExtractFuncType = Callable[
    [List[NoteEvent], int, int],  # input
    Tuple[List[np.ndarray], List[np.ndarray]],  # output
]


def extract_windows_single(events, window_size, step_size) -> tuple[list, list]:
    """Extract windows and labels from a list of note events, single label per window"""
    windows, labels = [], []
    for i in range(0, len(events) - window_size, step_size):
        window = events[i : i + window_size]
        windows.append(preprocess_window(window))
        n: NoteEvent = window[window_size // 2]
        labels.append(0 if n.hand == "left" else 1)
    return windows, labels


def extract_windows_seq_2_seq(events, window_size, step_size) -> tuple[list, list]:
    """Extract windows and labels from a list of note events, output per timestep"""
    windows, labels = [], []
    for i in range(0, len(events) - window_size, step_size):
        window = events[i : i + window_size]
        windows.append(preprocess_window(window))
        labels.append(np.array([0 if n.hand == "left" else 1 for n in window]))
    return windows, labels


@memory.cache
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


def k_fold_split(k):
    paths = list(Path("data").rglob("*.mid"))
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


def train_loop(
    model, train_loader, val_loader, num_epochs, optimizer, criterion, device, logger
):
    # Store metrics for all epochs
    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Function to process batches
    def process_batch(windows, labels):
        windows = windows.to(device)
        labels = labels.unsqueeze(1).float().to(device)
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

    # Loop over epochs
    for epoch in range(num_epochs):
        model.train()
        train_losses, train_accs = [], []
        for windows, labels in train_loader:
            loss, y_t, y_p = process_batch(windows, labels)
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
                loss, y_t, y_p = process_batch(windows, labels)
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

    return metrics
