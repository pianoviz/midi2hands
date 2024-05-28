# midi2hands

This project is about training and evaluating deep learning models for the task of predicting what hand is supposed to play what note in a piano piece.

### Repo layout

├── config
├── data
│   ├── test
│   └── train
├── good_results
├── results
├── models
├── notebooks
├── requirements.txt
├── train.py
├── utils.py

The idea is that models are defined in separate files as well as configurations.
utils.py contain all helper functions and is used by train.py.
Configurations of the models are passed to train.py as .json files.
Results, model weights and logs are stored results/.

Notebooks have been used to test and iterate and are not needed to rerun our results.
