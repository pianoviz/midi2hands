import argparse
from pathlib import Path

from midiutils.midi_preprocessor import MidiPreprocessor
from midiutils.types import NoteEvent

from midi2hands.models.generative import GenerativeHandFormer
from midi2hands.models.onnex.onnex_model import ONNXModel


def main(path: Path) -> list[NoteEvent]:
  model = ONNXModel()
  handformer = GenerativeHandFormer(model=model)
  events = MidiPreprocessor().get_midi_events(path)
  events_labeled, _, _ = handformer.inference(events=events, window_size=model.window_size, device="cpu")

  return events_labeled


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process a MIDI file and output labeled NoteEvents.")
  parser.add_argument("--input", type=Path, help="Path to the input MIDI file.")
  parser.add_argument("--output", type=Path, help="Path to the output JSON file.")

  args = parser.parse_args()

  events_labled = main(path=args.input)

  # # Write output to JSON
  # with open(args.output_path, "w") as output_file:
  #   output_file.write(events_labled)
