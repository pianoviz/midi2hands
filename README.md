# midi2hands

This project is about training and evaluating deep learning models for the task
of predicting what hand is supposed to play what note in a piano piece.
There are two main architectures used in this project:

1.  LSTM
2.  Transformer

The models can also be trained in two different ways,

1. Generative: Auto-regressive models that predict the next note given the
   previously predicted notes.
2. Discriminative: Models that predict the hand of the note given the note.

This is discussed in more detail in the [report](mid2hands.pdf).

This project is integrated in
[midi2vid](https://github.com/oscaraandersson/midi2vid), a tool that can
generate videos of piano performances from midi files.

## Installation
For inference, you can install the package with pip:
```bash
pip install midi2hands
```

For training, you can clone the repository and install the package with pip:
```bash
git clone ...
cd midi2hands
pip install -e .[train]
```

## Usage

Here is an example for inference:
```bash
midi2vid -i src/midi2hands/data/test/000-Faure_ClaireDeLune_Op46No2.mid
```

You can also train new models by running train.py. The models can be configured
from the script. Here is an example:
```bash
python src/midi2hands/train.py
```

## Data

The data is included in the repository. The data comes from the following
project, [hannds](https://github.com/cemfi/hannds).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.


