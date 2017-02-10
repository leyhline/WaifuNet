#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load existing model with trained weights and evaluate
testset data.

Returns a matrix where the rows are the predictions and the
columns are the actual categories.

Created on Wed Jan 25 21:05:08 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""


import numpy as np
from src.model import SimpleConvNet
from src.trainingset import TrainingSet

TEST_SAMPLES   = 200 * 100
BATCH_DIVIDER = 2  # Hard to explain... If this one is bigger 
                   # the batch size will become smaller.
QUERY_SIZE = 10
MAPPING = np.array(("Dress", "Nude", "School Uniform", "Swimsuit"),
                    dtype=np.unicode)


def test():
    testset = TrainingSet()
    testset.initialize("deeplearning/testset", "deeplearning/testset_txt",
                    batch_divider=BATCH_DIVIDER)
    model = SimpleConvNet()
    model.load_weights("train.hdf5")
    result = np.zeros((4, 4), dtype=np.int32)
    s = 0
    while s < TEST_SAMPLES:
        img, val = next(testset.training)
        preds = model.predict_classes(img, verbose=0)
        val = list(map(np.argmax, val))
        assert len(preds) == len(val)
        for i in range(len(preds)):
            result[preds[i], val[i]] += 1
        s += len(preds)
    return result


if __name__ == "__main__":
    result = test()
    print(result)
