# midi2hands

This project is about training and evaluating deep learning models for the task of predicting what hand is supposed to play what note in a piano piece.

### Repo layout
.
├── README.md
├── archive
│   └── train_lstm.py
├── config
│   ├── discriminative_lstm.json
│   ├── discriminative_transformer.json
│   ├── generative_lstm.json
│   └── generative_transformer.json
├── data
│   ├── test
│   └── train
├── good_results
│   ├── ancient_tesla_41.json
│   ├── nostalgic_bohr.json
│   ├── ownderful_neumann_68.json
│   ├── quiet_lovelace_92.json
│   └── results.json
├── models
│   ├── handformer.py
│   └── lstm.py
├── notebooks
│   ├── lstm.ipynb
│   └── transformer.ipynb
├── requirements.txt
├── results
│   └── quiet_bohr_90...
│   └── etc...
├── train.py
├── utils.py
├── eval.py
└── venv


The idea is that models are defined in separate files as well as configurations.
utils.py contain all helper functions and is used by train.py and eval.py.

Configurations of the models are passed to train.py as .json files.
Results, model weights and logs from training are stored results/.
The directory good_results/ contain the results that were used in the submitted project report.

eval.py is used in conjunction with a directory that holds the model to be evaluated.


Notebooks have been used to test and iterate and are not needed to rerun our results.
