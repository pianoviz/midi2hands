import unittest

from midiutils.types import NoteEvent

from midi2hands.models.discriminatory import DiscriminatoryHandFormer
from midi2hands.models.generative import GenerativeHandFormer
from midi2hands.models.mock.model_mock import ModelMock


class TestInference(unittest.TestCase):
  def setUp(self):
    model = ModelMock()
    self.generative = GenerativeHandFormer(model=model)
    self.discriminatory = DiscriminatoryHandFormer(model=model)

    self.events = [
      NoteEvent(note=23, velocity=45, start=40, end=500),
      NoteEvent(note=68, velocity=45, start=600, end=1000),
      NoteEvent(note=23, velocity=45, start=1200, end=2000),
    ]

  def test_generative_model_inference(self):
    # setup

    estimated_events, y_true, y_pred = self.generative.inference(events=self.events, window_size=30, device="cpu")

    assert len(estimated_events) == len(self.events)
    assert len(y_true) == len(self.events)
    assert len(y_pred) == len(self.events)

  def test_discriminatory_model_inference(self):
    # setup

    estimated_events, y_true, y_pred = self.generative.inference(events=self.events, window_size=30, device="cpu")

    assert len(estimated_events) == len(self.events)
    assert len(y_true) == len(self.events)
    assert len(y_pred) == len(self.events)
