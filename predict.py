#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 21:24:39 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""


import sys
from src.model import SimpleConvNet
import cv2
import numpy as np


MAPPING = np.array(("Dress", "Japanese Clothes", "Nude",
                    "School Uniform", "Shirt", "Swimsuit"),
                    dtype=np.unicode)
TARGET_SIZE = 200
FEATURE_DETECTOR = cv2.AKAZE_create()


def crop(img):
    """
    Crop image to make it quadratic using a feature detector for finding a nice center.
    (I know, feature detectors are not for something like this...)
    """
    kp = FEATURE_DETECTOR.detect(img)
    ysize, xsize = img.shape[:2]
    smallest = min(xsize, ysize)
    if not kp:
        print("No keypoints. Crop image in the center.")
        if xsize > ysize:
            dst = img[:, xsize // 2 - smallest // 2:xsize // 2 + smallest // 2]
        else:
            dst = img[xsize // 2 - smallest // 2:xsize // 2 + smallest // 2,:]
        return dst
    kp_sum = np.zeros(abs(xsize - ysize))
    if xsize > ysize:
        x_or_y = 0
    else:
        x_or_y = 1
    for i in range(kp_sum.size):
        for k in kp:
            if k.pt[x_or_y] - i < smallest:
                kp_sum[i] += k.response
    imax = kp_sum.argmax()
    if x_or_y:
        dst = img[imax:imax+smallest,:]
    else:
        dst = img[:,imax:imax+smallest]
    return dst


def preprocess(img):
    """Preprocesses given image. Using crop and resize if necessary.
    Also adds another dimension."""
    ysize, xsize = img.shape[:2]
    if xsize < TARGET_SIZE or ysize < TARGET_SIZE:
        print("Image size too small: {} x {}".format(xsize, ysize))
        return
    # Resize image.
    if xsize > ysize:
        xsize = round(xsize / ysize * TARGET_SIZE)
        ysize = TARGET_SIZE
    elif xsize < ysize:
        ysize = round(ysize / xsize * TARGET_SIZE)
        xsize = TARGET_SIZE
    else:
        xsize = ysize = TARGET_SIZE
    dst = cv2.resize(img, (xsize, ysize), interpolation=cv2.INTER_AREA)
    # Crop image if it is not already quadratic.
    if not xsize == ysize:
        dst = crop(dst)
    dst = np.expand_dims(dst, axis=0)
    return dst


def predict(img):
    """Use neural network to predict class of image."""
    model = SimpleConvNet()
    model.load_weights("train.hdf5")
    return model.predict_classes(img)
    

def print_classes(preds):
    """Prints the filename and the predicted class."""
    preds = map(lambda x: MAPPING[x], preds)
    preds = list(preds)
    for i in range(len(filenames)):
        print(filenames[i], preds[i], sep=": ")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} image [ images ... ]".format(__file__), file=sys.stderr)
        sys.exit(1)
    else:
        filenames = sys.argv[1:]
    images = np.empty((len(filenames), TARGET_SIZE, TARGET_SIZE, 3))
    for i in range(len(filenames)):
        img = cv2.imread(filenames[i])
        img = preprocess(img)
        images[i] = img
    preds = predict(images)
    print_classes(preds)