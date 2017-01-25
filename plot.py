#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple script for plotting a csv file from keras training
with loss and accuracy for training- and validation set.

File should look like the following:
    epoch,acc,loss,val_acc,val_loss
    0,0.391549987141,1.52331774217,0.463329985239,1.37888392267
    ...

Created on Wed Jan 25 19:55:12 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""


import matplotlib.pyplot as plt
import sys

FILENAME = "logs/train.log"


if __name__ == "__main__":
    if len(sys.argv) == 1:  # Defaults to FILENAME.
        filename = FILENAME
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        print("Usage: {} [ csv-file ]".format(__file__), file=sys.stderr)
        sys.exit(1)        
    plt.plotfile(filename, cols=range(0,5), subplots=False)
    plt.show()