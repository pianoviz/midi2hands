from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from joblib import Memory
from midiutils.midi_preprocessor import MidiPreprocessor
from midiutils.types import NoteEvent
from numpy.typing import NDArray

memory = Memory(location="cache", verbose=0)


class HandModel(ABC):
  @abstractmethod
  def __call__(self, x: NDArray[np.float32]) -> list[float]: ...

  @property
  @abstractmethod
  def model(self) -> Any: ...

  @property
  @abstractmethod
  def window_size(self) -> int: ...


class HandFormer(ABC):
  """Estimate which hand play a note.

  There are two main ways to preprocess the data, generative and discriminatory.
  The windowing is different in the two methods.
  """

  def convert_hand_to_number(self, hand: str | None):
    return 0 if hand == "left" else (1 if hand == "right" else -1)

  @abstractmethod
  def inference(
    self,
    events: list[NoteEvent],
    window_size: int,
    device: str,
  ) -> Tuple[list[NoteEvent], list[int], list[int]]: ...

  @abstractmethod
  def preprocess_window(self, note_events: list[NoteEvent]) -> NDArray[np.float32]: ...

  @abstractmethod
  def extract_windows(self, events: list[NoteEvent], window_size: int) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
    """Extract windows from a number of note events.
    Each window has a label.
    Returns ([X1, X2, ...], [y1, y2, ...]) where X is the window and y is the
    label
    """
    ...

  def extract_windows_from_files(self, paths: list[Path], window_size: int) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    all_windows: list[NDArray[np.float32]] = []
    all_labels: list[NDArray[np.float32]] = []
    mp = MidiPreprocessor()
    for path in paths:
      events = mp.get_midi_events(path)
      windows, labels = self.extract_windows(events, window_size)
      all_windows.extend(windows)
      all_labels.extend(labels)
    return np.array(all_windows), np.array(all_labels)

  def _pad_events(self, events: list[NoteEvent], window_size: int) -> list[NoteEvent]:
    """Pad the events with None values at the beginning and the end of the list.

    Args:
    events (List[NoteEvent]): List of note events.
    window_size (int): The size of the window for which to pad the events.

    Returns:
    List[NoteEvent]: New list of note events with padding added.

    """
    # Calculate the amount of padding needed on each side
    m = window_size // 2
    padded_events: list[NoteEvent] = []
    for _ in range(m):
      dummy_note = NoteEvent(note=-1, velocity=-1, start=-1, hand=None)
      dummy_note.set_end(-1)
      padded_events.append(dummy_note)
    padded_events.extend(events)
    # Create and add padding at the end of the list
    for _ in range(m):
      dummy_note = NoteEvent(note=-1, velocity=-1, start=-1, hand=None)
      dummy_note.set_end(-1)
      padded_events.append(dummy_note)
    return padded_events
