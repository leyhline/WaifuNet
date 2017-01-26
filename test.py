#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load existing model with trained weights and evaluate
testset data.

Created on Wed Jan 25 21:05:08 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""


import numpy as np
from src.model import SimpleConvNet
from src.trainingset import TrainingSet


TEST_SAMPLES   = 100000
BATCH_DIVIDER = 4  # Hard to explain... If this one is bigger 
                   # the batch size will become smaller.
QUERY_SIZE = 10 * BATCH_DIVIDER
MAPPING = np.array(("Dress", "Japanese Clothes", "Nude",
                    "School Uniform", "Shirt", "Swimsuit"),
                    dtype=np.unicode)


def test():
    testset = TrainingSet()
    testset.initialize("deeplearning/testset", "deeplearning/testset_txt",
                    batch_divider=BATCH_DIVIDER)
    model = SimpleConvNet()
    model.load_weights("train.hdf5")
    correct = np.zeros(6, dtype=np.int32)
    incorrect = np.zeros((6, 6), dtype=np.int32)
    i = 0
    print(i, TEST_SAMPLES, sep="/")
    while i < TEST_SAMPLES:
        img, val = next(testset.training)
        preds = model.predict_classes(img)
        val = map(np.argmax, val)
        assert len(preds) == len(val)
        for i in range(len(preds)):
            if preds[i] == val[i]:
                correct[preds[i]] += 1
            else:
                incorrect[preds[i], val[i]] += 1
        i += len(preds)
        print(i, TEST_SAMPLES, sep="/")
    return correct, incorrect


if __name__ == "__main__":
    # TODO What is the output? Where to get it from?
    correct, incorrect = test()
    print(MAPPING)
    print("Correctly predicted:")
    print(correct)
    print("Incorrectly predicted:")
    print(incorrect)
