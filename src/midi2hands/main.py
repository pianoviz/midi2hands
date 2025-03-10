import argparse
from pathlib import Path

from midiutils.midi_preprocessor import MidiPreprocessor

from midi2hands.models.generative import GenerativeHandFormer
from midi2hands.models.onnex.onnex_model import ONNXModel


def main():
  parser = argparse.ArgumentParser(description="Process a MIDI file and output labeled NoteEvents.")
  parser.add_argument("-i", "--input", type=Path, help="Path to the input MIDI file.")
  args = parser.parse_args()

  model = ONNXModel()
  handformer = GenerativeHandFormer(model=model)
  events = MidiPreprocessor().get_midi_events(midi_path=args.input)
  events_labeled, _, _ = handformer.inference(events=events, window_size=model.window_size, device="cpu")
  for event in events_labeled:
    print(event.hand)


if __name__ == "__main__":
  main()
