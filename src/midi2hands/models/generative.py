from copy import deepcopy

import numpy as np
from midiutils.types import NoteEvent
from numpy.typing import NDArray

from midi2hands.models.interface import HandFormer, HandModel


class GenerativeHandFormer(HandFormer):
  def __init__(
    self,
    model: HandModel,
  ):
    self.model = model

  def inference(self, events: list[NoteEvent], window_size: int, device: str) -> tuple[list[NoteEvent], list[int], list[int]]:
    padded_events = self._pad_events(events=deepcopy(events), window_size=window_size)
    y_true: list[int] = []
    y_pred: list[int] = []
    h = window_size // 2
    for i in range(h, len(events) + h, 1):
      window_events = padded_events[i - h : i + h]
      preprocessed_window = self.preprocess_window(window_events)
      preprocessed_window[:, -1] = -1  # we don't know the output yet
      prev_out = y_pred[-h:]
      preprocessed_window[: len(prev_out), -1] = prev_out

      label = padded_events[i].hand
      label = self.convert_hand_to_number(label)
      y_true.append(label)

      output = self.model(np.expand_dims(preprocessed_window, axis=0))
      output = float(output[0])
      output = 0 if output < 0.5 else 1
      y_pred.append(output)
    return events, y_true, y_pred

  def preprocess_window(self, note_events: list[NoteEvent]) -> NDArray[np.float32]:
    """Convert the list of notes to a numpy array.

    also normalize the start times"""

    def convert(n: NoteEvent):
      return (n.note, n.start, n.end, self.convert_hand_to_number(n.hand))

    window = np.array([convert(n) for n in note_events], dtype=np.float32)
    non_pad_indices = np.where(window[:, 0] != -1)[0]
    window[non_pad_indices, 1:3] = window[non_pad_indices, 1:3] / np.max(window[non_pad_indices, 2])
    window[non_pad_indices, 0] = (window[non_pad_indices, 0] - 21) / 88
    return window

  def extract_windows(self, events: list[NoteEvent], window_size: int) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
    """Extract windows and labels from a list of note events
    Include the label in the
    """
    # print(f"window size: {window_size}")
    windows: list[NDArray[np.float32]] = []
    labels: list[NDArray[np.float32]] = []
    # print(f"events: {len(events)}")
    padded_events = self._pad_events(events=deepcopy(events), window_size=window_size)
    # print(f"padded events: {len(padded_events)}")
    h = window_size // 2
    for i in range(h, len(events) + h):
      window = padded_events[i - h : i + h]
      # print(f"size: {len(window)}")
      preprocessed_window = self.preprocess_window(window)
      # print(f"size: {preprocessed_window.shape}")
      label = self.convert_hand_to_number(window[h].hand)
      label = np.array([label])
      for j in range(h, window_size):
        preprocessed_window[j, -1] = -1
      windows.append(preprocessed_window)
      labels.append(label)
    return windows, labels
