# -*- coding: utf-8 -*-
"""
This class initialized generators for indefinetly iterating over training 
and (if specified) validation data.
This is necessary for the fit_generator-method in keras models.

At first the data is loaded into memory.
To save memory the images are stored in encoded form and
are decoded into numpy arrays on the fly if necessary.

Created on Wed Jan 11 07:52:00 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""

import cv2
import numpy as np
import logging
import tarfile
from io import BytesIO
from itertools import cycle
from os.path import splitext
from concurrent.futures import ThreadPoolExecutor
from getpass import getpass
from .pcloud import PCloud
from keras.preprocessing.image import ImageDataGenerator


CATEGORIES = ("dres", "nude", "scho", "swim")
ARCHIVE_NAMES = ("training.tar", "validation.tar")


class TrainingSet:
    """
    Class for getting generators to iterate over training examples.
    These are located at a cloud storage service and are loaded dynamically.
    """
    
    log = logging.getLogger("trainingset")
    datagen = ImageDataGenerator(rotation_range=30,
                                 width_shift_range=.2,
                                 height_shift_range=.2,
                                 shear_range=.2,
                                 zoom_range=.2,
                                 horizontal_flip=True,
                                 fill_mode="nearest",
                                 rescale=1./255)
    
    def __init__(self, username=None, password=None):
        """Ask for username/password for PCloud access."""
        print("Logging into pCloud account.")
        if not username or not password:
            username = input(prompt="Username: ")
            password = getpass(prompt="Password: ")
        self.cloud = PCloud(username, password)
        print("Success!")
        # You need to call initialize to use these.
        self.training = None
        self.validation = None
        self.testset = None

    def initialize(self, filenames=ARCHIVE_NAMES, workers=4):
        """
        Load training and validation examples into memory.
        Initialize generators for iterating over training examples.
        There have to be folders of the same name as the CATEGORIES
        in the root directory of the cloud.
        """
        files_per_category = map(self.cloud.get_files_in_folder, CATEGORIES)
        filter_func = lambda x: filter(lambda y: y[0] in filenames, x)
        files_per_category = map(filter_func, files_per_category)
        with ThreadPoolExecutor(max_workers=workers) as e:
            data = e.map(self._get_data, files_per_category)
        self.data = dict(zip(CATEGORIES, data))
            
    def _get_data(self, files, max_size_per_file=1073741824):
        """
        Input is an iterable of (filename, fileid) tuples.
        Downloads and extracts the data and returns them as binary data.
        Returns dictionary with file_basename:data_list
        """
        filenames, fileids = zip(*files)
        filenames = (splitext(filename)[0] for filename in filenames)
        arg = cycle((max_size_per_file,))
        requests = map(self.cloud.get_file, fileids, arg)
        data = map(self._extract, requests)
        return dict(zip(filenames, data))
        
    def _extract(self, request):
        """
        Takes a raw request object of a tarfile (uncompressed) and extracts 
        its content.
        Returns a list of its objects as binary data.
        """
        binary_data = []
        with BytesIO(request.read()) as stream:
            with tarfile.open(mode="r:", fileobj=stream) as tf:
                for member in tf:
                    with tf.extractfile(member) as mf:
                        binary_data.append(mf.read())
        return binary_data
        
    def _decode_image(self, data):
        """
        Takes raw image data and decodes it to a numpy array.
        """
        raw_array = np.frombuffer(data, dtype=np.int8)
        return cv2.imdecode(raw_array, cv2.IMREAD_COLOR)
