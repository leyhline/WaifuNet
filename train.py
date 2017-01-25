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
import logging
import logging.config
import yaml


TRAINING_SAMPLES   = 300000
VALIDATION_SAMPLES = 100000
EPOCHES = 6
INITIAL_EPOCH = 3
BATCH_DIVIDER = 4  # Hard to explain... If this one is bigger 
                   # the batch size will become smaller.
QUERY_SIZE = 10 * BATCH_DIVIDER
VERBOSE = 1

# Load and configure logging.
with open("logging.yaml") as f:
    logging_config = yaml.load(f)
logging.config.dictConfig(logging_config)


def train():
    tset = TrainingSet()
    tset.initialize("deeplearning/training", "deeplearning/training_txt",
                    "deeplearning/validation", "deeplearning/validation_txt",
                    batch_divider=BATCH_DIVIDER)
    model = SimpleConvNet()
    sgd = SGD()
    model.compile(sgd,
                  "categorical_crossentropy",
                  metrics=["accuracy"])
    if INITIAL_EPOCH:
        print("Loading model weights: train.hdf5")
        model.load_weights("train.hdf5")
        print("Resume training.")
        epoches = EPOCHES - INITIAL_EPOCH + 1
    else:
        print("Starting training.")
        epoches = EPOCHES
    history = model.fit_generator(
                        tset.training,
                        TRAINING_SAMPLES,
                        epoches,
                        verbose=VERBOSE,
                        callbacks=[ModelCheckpoint("train.hdf5",
                                                   save_weights_only=True), 
                                   CSVLogger("logs/train.log", append=True)],
                        validation_data=tset.validation,
                        nb_val_samples=VALIDATION_SAMPLES,
                        max_q_size=QUERY_SIZE)
    return model, history


if __name__ == "__main__":
    train()
