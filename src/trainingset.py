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


import numpy as np
import logging
import tarfile
from io import BytesIO
from itertools import cycle
from os.path import splitext
from concurrent.futures import ThreadPoolExecutor
from getpass import getpass
from .pcloud import PCloud
from .preprocessing import ImageDataGenerator


CATEGORIES = ("dres", "nude", "scho", "swim")
ARCHIVE_NAMES = ("training.tar", "validation.tar")
AUGMENT = {"training":True, "validation":False}


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
    # Image data genegator which does no augmentation.
    datagen_noaug = ImageDataGenerator()
    
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

    def initialize(self, filenames=ARCHIVE_NAMES, batch_size=50, workers=4,
                   augment=AUGMENT):
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
        data = dict(zip(CATEGORIES, data))
        self.data = self._init_generators(data, batch_size, filenames=filenames,
                                          augment=augment)

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

    def _init_generators(self, data, batch_size, filenames, augment):
        """
        Takes a dictionary of categories:data and returns as many
        indefinite imagegenerators as there are ARCHIVE_NAMES
        as a dictionary.
        """
        sorted_data = {}
        basenames = (splitext(filename)[0] for filename in filenames)
        # Reorder dict of dicts to filename:categories structure.
        for filename in basenames:
            sorted_data[filename] = ((cat, data[cat][filename]) for cat in CATEGORIES)
        datagens = {}
        for name in sorted_data:
            X = []
            y = []
            for cat, bdatas in sorted_data[name]:
                cat = self._category_to_array(cat)
                for bdata in bdatas:
                    X.append(bdata)
                    y.append(cat)
            # TODO: This is not very robust if the AUGMENT dict is incomplete.
            if augment[name]:
                datagens[name] = self.datagen.flow(X, (200, 200, 3), y, batch_size)
            elif not augment[name]:
                datagens[name] = self.datagen_noaug.flow(X, (200, 200, 3), y,
                                        batch_size, shuffle=False)
        return datagens                
        
    def _category_to_array(self, category):
        """
        Takes a category name as a string and returns a binary numpy array.
        """
        narray = np.zeros(len(CATEGORIES), dtype=np.bool)
        narray[CATEGORIES.index(category)] = True
        return narray