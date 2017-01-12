#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 20:59:25 2017

@author: Thomas Leyh
"""

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense


def WaifuVGG16():
    """Customize VGG16 model a bit to fit own data."""
    model = Sequential()
    model.add(VGG16(include_top=False,
                   weights=None,
                   input_shape=(200,200,3)))
    model.add(Flatten(name="flatten"))
    model.add(Dense(256, activation="relu", name="fc1"))
    model.add(Dense(256, activation="relu", name="fc2"))
    model.add(Dense(6, activation="softmax", name="predictions"))
    return model