#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 21:52:15 2017

@author: Thomas Leyh
"""

from src.model import WaifuVGG16
from src.trainingset import TrainingSet
from keras.callbacks import ModelCheckpoint, CSVLogger
import sys


TRAINING_SAMPLES =   300000
VALIDATION_SAMPLES = 100000
EPOCHES=5
QUERY_SIZE=100
WORKERS=4  # Only relevant when pickle_save is set in fit_generator()


def train():
    tset = TrainingSet()
    tset.initialize("deeplearning/training", "deeplearning/training_txt",
                    "deeplearning/validation", "deeplearning/validation_txt",
                    batch_divider=10)
    model = WaifuVGG16()
    model.compile("sgd",
                  "categorical_crossentropy",
                  metrics=["categorical_crossentropy"])
    print("Starting training.")
    history = model.fit_generator(
                        tset.training,
                        TRAINING_SAMPLES,
                        EPOCHES,
                        verbose=1,
                        callbacks=[ModelCheckpoint("train.hdf5",
                                                   save_weights_only=True), 
                                   CSVLogger("train.log", append=False)],
                        validation_data=tset.validation,
                        nb_val_samples=VALIDATION_SAMPLES,
                        max_q_size=QUERY_SIZE
                        pickle_safe=False,
                        nb_workers=WORKERS)
    return model, history


if __name__ == "__main__":
    q = ''
    while q != 'y' or q != 'n':
        q = input("Start training model? Hope you got enough time at hand. y/n ")
        if q == 'y':
            train()
        if q == 'n':
            sys.exit()
