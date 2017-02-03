# -*- coding: utf-8 -*-
"""
This class initialized generators for indefinetly iterating over training 
and (if specified) validation data.
This is necessary for the fit_generator-method in keras models.

The data is aquired from pCloud during training. Because of simple multithreading
there is hopefully not that much delay so most of the data should be
available without much training delay.

All data is held in memory. There is no need to save files.

Created on Wed Jan 11 07:52:00 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""

import cv2
import numpy as np
import logging
import random
from os.path import splitext
from concurrent.futures import ThreadPoolExecutor
from getpass import getpass
from .pcloud import PCloud


def image_preprocessing(img):
    """Some image preprocessing for easier training."""
    # From keras.applications.imagenet_utils
    # Zero-center by mean pixel
    # Subtract BGR mean of all training images. (calculated previously)
    img[:, :, :, 0] -= 173.40410352
    img[:, :, :, 1] -= 177.05503129
    img[:, :, :, 2] -= 192.17966982
    # Flip image tiles randomly on horizontal axis.
    for tile in img:
        if random.getrandbits(1):
            cv2.flip(tile, 1, tile)


class TrainingSet:
    """
    Class for getting generators to iterate over training examples.
    These are located at a cloud storage service and are loaded dynamically.
    """
    
    log = logging.getLogger("trainingset")
    
    def __init__(self):
        """Ask for username/password for PCloud access."""
        print("Logging into pCloud account.")
        print("Username:", end=" ")
        username = input()
        password = getpass()
        self.cloud = PCloud(username, password)
        print("Success!")
        # These are initialized with the corresponding method:
        self.training = None
        self.validation = None

    def initialize(self, input_folder, target_folder,
                   validation_folder=None, validation_target_folder=None,
                   batch_divider=0):
        """
        Initialize generators for iterating over training examples
        and if specified also get generator for validation examples.
        Arguments take the path/to/folder in cloud storage.
        
        Standard batch size if 100. The batch_divider is some value
        between 0 and 100. A larger value reduces the batch size.
        """
        x_files, y_files = self._filelist(input_folder, target_folder)
        x_name, x_fid = zip(*x_files)
        y_name, y_fid = zip(*y_files)
        self.images = tuple(zip(x_name, self._load_into_memory(x_fid)))
        self.targets = tuple(zip(y_name, self._load_into_memory(y_fid)))
        self.training = self._create_generator(self.images, self.targets,
                                               batch_divider)
        if validation_folder and validation_target_folder:
            x_vali, y_vali = self._filelist(validation_folder, 
                                            validation_target_folder)
            x_name_val, x_fid_val = zip(*x_vali)
            y_name_val, y_fid_val = zip(*y_vali)
            self.images_val = tuple(zip(x_name_val, self._load_into_memory(x_fid_val)))
            self.targets_val = tuple(zip(y_name_val, self._load_into_memory(y_fid_val)))
            self.validation = self._create_generator(self.images_val, self.targets_val,
                                                     batch_divider)
            
    def _filelist(self, input_folder, target_folder):
        """
        Get a 2-entry list of the files from the cloud storage.
        Important: The inner lists contain tuples (filename, fileid).
        """
        files = (self.cloud.get_files_in_folder(*folder.split("/"))
                 for folder in (input_folder, target_folder))
        # Sort the file lists.
        files = (sorted(flist, key=lambda x: int(splitext(x[0])[0]))
                 for flist in files)
        files = tuple(files)
        assert len(files[0]) == len(files[1])
        # Check if you have fitting input and target file pair.
        checkpairs = map(lambda inpt, trgt: inpt[0][:-5] == trgt[0][:-4],
                         files[0], files[1])
        for check in checkpairs:
            assert check, "Training and target data does not fit together."
        return files
        
    def _create_generator(self, imagedata, targetdata,
                          batch_div=0, img_per_file=100):
        """
        Returns a generator who holds (x, y) tuples where each tuple
        is a training batch (here: 100 images).
        The generator runs indefinetly over the data.
        batch_div divides the standard batch size (100) by its value.
        img_per_file only needs to be given if batch_div is set.
        """
        if batch_div:
            batch_div = self._find_divider(batch_div, img_per_file)
            batch_size = img_per_file // batch_div
        image_generator = self._data_from_memory(imagedata, self._raw_to_images)
        target_generator = self._data_from_memory(targetdata, self._raw_to_array)
        while True:
            image_name, inputs = next(image_generator)
            target_name, targets = next(target_generator)
            # Check if you really got the right image-target combo.
            self.log.info("Files {} and {} received.".format(image_name, target_name))
            assert image_name[:-5] == target_name[:-4], "{} does not fit to {}.".format(image_name, target_name)
            image_preprocessing(inputs)
            if batch_div:
                for j in range(0, img_per_file, batch_size):
                    yield (inputs[j:j+batch_size],
                           targets[j:j+batch_size])
            else:
                yield inputs, targets
    
    def _find_divider(self, divider, divisor):
        """Returns a divider without remainder for a batch (divisor).
           The retuned divider is >= divider argument."""
        for i in range(divider, 0, -1):
            if divisor % divider == 0:
                return divider
            else:
                divider += 1
    
    def _load_into_memory(self, fileid, workers=8):
        """Load all the training data into RAM because it too slow
           to download everything in every epoch."""
        with ThreadPoolExecutor(max_workers=workers) as e:
            data = tuple(e.map(self._load_into_memory_helper, fileid))
        # TODO: It may be better to return a zip object.
        return data
            
    def _load_into_memory_helper(self, fileid):
        """Get file from fileid and return its raw binary data."""
        self.log.info("Downloading file with id: {}".format(fileid))
        return self.cloud.get_file(fileid, 716800).read()
    
    def _data_from_memory(self, data, processing):
        """Generator for loading all data into memory once and then
           indefinetly iterating over it."""
        i = 0
        while True:
            self.log.info("Pop element {}".format(data[i][0]))
            yield data[i][0], processing(data[i][1])
            if i == len(data) - 1:
                i = 0
            else:
                i += 1
        
    def _raw_to_images(self, raw, xtiles=10, ytiles=10):
        """Take raw data (actually it's a class from requests package)
           and decode it to a batch of image arrays. (4 dimensions)"""
        raw_array = np.frombuffer(raw, dtype=np.int8)
        montage = cv2.imdecode(raw_array, cv2.IMREAD_COLOR)
        ysize = montage.shape[0]
        xsize = montage.shape[1]
        # Hardcoded check if this dataset fits to model.
        assert xsize == 2000
        assert ysize == 2000
        ytilesize = ysize // ytiles
        xtilesize = xsize // xtiles
        images = np.empty((xtiles*ytiles, ytilesize, xtilesize, 3),
                          dtype=np.float32)
        for i in range(xtiles * ytiles):
            yfrom = (i * xtilesize // xsize) * ytilesize
            yto = yfrom + ytilesize
            xfrom = i * xtilesize % xsize
            xto = xfrom + xtilesize
            image = montage[yfrom:yto, xfrom:xto]
            images[i] = image
        return images
        
    def _raw_to_array(self, raw, lines=100):
        """Take raw data (actually it's a class from requests package)
           and decode it to a batch of binary arrays for classification. (4 dimensions)"""
        mapping = {"nude":0, "scho":1, "swim":2}
        arrays = np.empty((lines, len(mapping)), dtype=np.bool)
        i = 0
        for line in raw.decode().split("\n"):
            if not line:
                continue
            line = line[2:6]
            array = np.zeros(len(mapping), dtype=np.bool)
            array[mapping[line]] = True
            arrays[i] = array
            i += 1
        # Assert correct batch size.
        assert i == lines
        return arrays
