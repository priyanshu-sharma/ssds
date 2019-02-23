##This is lisa_config.py which is used to store the configuration such as file paths, and class labels, etc.

import os

BASE_PATH = "lisa"
ANNOT_PATH = os.path.sep.join([BASE_PATH, "allAnnotations.csv"])

TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, "records/testing.record"])

TEST_SIZE = 0.25

CLASSES = {"pedestrianCrossing": 1, "signalAhead": 2, "stop": 3}
