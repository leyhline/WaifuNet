#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple script for building and training a model.
Just run train() or run the file from command line.
For me the training time per epoch is around 5 hours.

You don't give arguments to the function but instead edit the
global variables at the top of the file to control configurations.

Created on Thu Jan 12 21:52:15 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""

from src.model import SimpleConvNet
from src.trainingset import TrainingSet
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import SGD
import os
import logging
import logging.config
import yaml


TRAINING_SAMPLES   = 260000
VALIDATION_SAMPLES = 20000
EPOCHES = 50
BATCH_SIZE=50
QUERY_SIZE = 10
VERBOSE = 1
LOG_FILE = "logs/train.log"
WEIGHTS_FILE = "train.hdf5"

# Load and configure logging.
with open("logging.yaml") as f:
    logging_config = yaml.load(f)
logging.config.dictConfig(logging_config)


def train():
    tset = TrainingSet()
    tset.initialize(batch_size=BATCH_SIZE)
    model = SimpleConvNet()
    sgd = SGD(lr=0.01)
    model.compile(sgd,
                  "categorical_crossentropy",
                  metrics=["accuracy"])
    if os.path.exists(WEIGHTS_FILE) and os.path.exists(LOG_FILE):
        print("Loading model weights: " + WEIGHTS_FILE)
        model.load_weights(WEIGHTS_FILE)
        log_last_line = os.popen("tail -n1 " + LOG_FILE).readline().split(",")
        try:
            initial = int(log_last_line[0])
        except ValueError as e:
            e.args = ("Error: Could not read logfile: " + LOG_FILE,)
            raise e
        print("Resume training from epoch {}.".format(initial))
    else:
        print("Starting training.")
        initial = 0
    history = model.fit_generator(
                        tset.data["training"],
                        TRAINING_SAMPLES,
                        EPOCHES,
                        verbose=VERBOSE,
                        callbacks=[ModelCheckpoint(WEIGHTS_FILE,
                                                   save_weights_only=True),
                                   CSVLogger(LOG_FILE, append=True)],
                        validation_data=tset.data["validation"],
                        nb_val_samples=VALIDATION_SAMPLES,
                        max_q_size=QUERY_SIZE,
                        initial_epoch=initial)
    return model, history


if __name__ == "__main__":
    train()
