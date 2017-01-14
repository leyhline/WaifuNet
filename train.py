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

from src.model import WaifuVGG16
from src.trainingset import TrainingSet
from keras.callbacks import ModelCheckpoint, CSVLogger
import sys


TRAINING_SAMPLES =   300000
VALIDATION_SAMPLES = 100000
EPOCHES=1
BATCH_DIVIDER=100  # Hard to explain... If this one is bigger 
                  # the batch size will become smaller.
QUERY_SIZE=10 * BATCH_DIVIDER
VERBOSE=0


def train():
    tset = TrainingSet()
    tset.initialize("deeplearning/training", "deeplearning/training_txt",
                    "deeplearning/validation", "deeplearning/validation_txt",
                    batch_divider=BATCH_DIVIDER)
    model = WaifuVGG16()
    model.compile("sgd",
                  "categorical_crossentropy",
                  metrics=["categorical_crossentropy"])
    print("Starting training.")
    history = model.fit_generator(
                        tset.training,
                        TRAINING_SAMPLES,
                        EPOCHES,
                        verbose=VERBOSE,
                        callbacks=[ModelCheckpoint("train.hdf5",
                                                   save_weights_only=True), 
                                   CSVLogger("train.log", append=False)],
                        validation_data=tset.validation,
                        nb_val_samples=VALIDATION_SAMPLES,
                        max_q_size=QUERY_SIZE)
    return model, history


if __name__ == "__main__":
    train()
