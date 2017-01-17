#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""

from src.trainingset import TrainingSet
import sys
import logging
import logging.config
import yaml


TRAINING_SAMPLES   = 300000
VALIDATION_SAMPLES = 100000
EPOCHES = 1
BATCH_DIVIDER = 5  # Hard to explain... If this one is bigger 
                   # the batch size will become smaller.
QUERY_SIZE = 10 * BATCH_DIVIDER
VERBOSE = 1

# Load and configure logging.
with open("logging.yaml") as f:
    logging_config = yaml.load(f)
logging.config.dictConfig(logging_config)


def test():
    tset = TrainingSet()
    tset.initialize("deeplearning/training", "deeplearning/training_txt",
                    "deeplearning/validation", "deeplearning/validation_txt",
                    batch_divider=BATCH_DIVIDER)
    i = 0
    while True:
        x, y = next(tset.training)
        print(i, x.shape, y.shape)
        i += 100 // BATCH_DIVIDER


if __name__ == "__main__":
    test()
