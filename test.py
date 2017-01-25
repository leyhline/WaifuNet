#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load existing model with trained weights and evaluate
testset data.

Created on Wed Jan 25 21:05:08 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""

from src.model import SimpleConvNet
from src.trainingset import TrainingSet
from keras.optimizers import SGD


TEST_SAMPLES   = 100000
BATCH_DIVIDER = 1  # Hard to explain... If this one is bigger 
                   # the batch size will become smaller.
QUERY_SIZE = 10 * BATCH_DIVIDER


def test():
    testset = TrainingSet()
    testset.initialize("deeplearning/testset", "deeplearning/testset_txt",
                    batch_divider=BATCH_DIVIDER)
    model = SimpleConvNet()
    sgd = SGD()
    model.compile(sgd,
                  "categorical_crossentropy",
                  metrics=["accuracy"])
    model.load_weights("train.hdf5")
    model.evaluate_generator(testset.training,
                             val_samples=TEST_SAMPLES,
                             max_q_size=QUERY_SIZE)
if __name__ == "__main__":
    test()
