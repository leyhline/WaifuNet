# -*- coding: utf-8 -*-
"""
This file contains some simple model(s) build for this specific dataset.
At the moment all models are just from keras.applications with just
the most necessary modifications.

Created on Thu Jan 12 20:59:25 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.regularizers import l2


def WaifuVGG16():
    """Customize VGG16 model a bit to fit own data."""
    model = Sequential()
    model.add(VGG16(include_top=False,
                   weights=None,
                   input_shape=(200,200,3)))
    # Add top layers for getting categories (here: 6).
    # Hope size 256 is okay for the dense layers.
    model.add(Flatten(name="flatten"))
    model.add(Dense(256, activation="relu", name="fc1"))
    model.add(Dense(256, activation="relu", name="fc2"))
    model.add(Dense(6, activation="softmax", name="predictions"))
    return model

def SimpleConvNet():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(200, 200, 3), activation="relu",
                            name="block1_conv1", init="glorot_normal"))
    model.add(MaxPooling2D(name="block1_pool"))
    model.add(Convolution2D(64, 3, 3, activation="relu", init="glorot_normal",
                            name="block2_conv1"))
    model.add(MaxPooling2D(name="block2_pool"))
    model.add(Convolution2D(128, 3, 3, activation="relu", init="glorot_normal",
                            name="block3_conv1"))
    model.add(MaxPooling2D(name="block3_pool"))
    model.add(Convolution2D(256, 3, 3, activation="relu", init="glorot_normal",
                            name="block4_conv1"))
    model.add(MaxPooling2D(name="block4_pool"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(256, activation="relu", name="fc1", init="glorot_normal"))
    model.add(Dropout(0.50, name="dropout1"))
    model.add(Dense(3, activation="softmax", init="glorot_normal",
                    name="predictions"))
    return model
