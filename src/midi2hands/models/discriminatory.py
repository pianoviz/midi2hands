from copy import deepcopy

import numpy as np
from midiutils.types import NoteEvent
from numpy.typing import NDArray

from midi2hands.models.interface import HandFormer, HandModel


class DiscriminatoryHandFormer(HandFormer):
  def __init__(self, model: HandModel):
    self.model = model

  def inference(self, events: list[NoteEvent], window_size: int, device: str) -> tuple[list[NoteEvent], list[int], list[int]]:
    padded_events = self._pad_events(deepcopy(events), window_size)
    y_true: list[int] = []
    y_pred: list[int] = []
    h = window_size // 2
    for i in range(h, len(events) + h, 1):
      window_events = padded_events[i - h : i + h]
      label = window_events[h].hand
      preprocessed_window = self.preprocess_window(window_events)
      label = self.convert_hand_to_number(label)
      y_true.append(label)

      # tensor_window = torch.tensor(
      # preprocessed_window).unsqueeze(0).to(device)
      output = self.model(preprocessed_window)
      # output = output.squeeze().cpu().detach().numpy()
      output = float(output[0])
      output = 0 if output < 0.5 else 1
      y_pred.append(output)
    return events, y_true, y_pred

  def preprocess_window(self, note_events: list[NoteEvent]) -> NDArray[np.float32]:
    """Convert a subset of the list of notes into a window.
    also normalize the start times.

    The window will be a numpy array of shape (n_events, 3)
    """

    def convert(n: NoteEvent):
      return (n.note, n.start, n.end)

    window = np.array([convert(n) for n in note_events], dtype=np.float32)
    non_pad_indices = np.where(window[:, 0] != -1)[0]
    window[non_pad_indices, 1:3] = window[non_pad_indices, 1:3] / np.max(window[non_pad_indices, 2])
    window[non_pad_indices, 0] = (window[non_pad_indices, 0] - 21) / 88
    return window

  def extract_windows(self, events: list[NoteEvent], window_size: int) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
    """Extract windows and labels from a list of note events
    Include the label in the
    """
    windows: list[NDArray[np.float32]] = []
    labels: list[NDArray[np.float32]] = []
    padded_events = self._pad_events(events=deepcopy(events), window_size=window_size)
    h = window_size // 2
    for i in range(h, len(events) + h, 1):
      window_events = padded_events[i - h : i + h]
      preprocessed_window = self.preprocess_window(window_events)
      label = np.array([self.convert_hand_to_number(window_events[h].hand)])
      windows.append(preprocessed_window)
      labels.append(label)
    return windows, labels
