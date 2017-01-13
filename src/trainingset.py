#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 07:52:00 2017

@author: Thomas Leyh
"""

import cv2
import numpy as np
from getpass import getpass
from .pcloud import PCloud


# TODO Look further what this does before you use it.
def image_preprocessing(img):
    """Seems it is necessary to subtract the mean of the RGB."""
    # From keras.applications.imagenet_utils
    # Zero-center by mean pixel
    img[:, :, :, 0] -= 103.939
    img[:, :, :, 1] -= 116.779
    img[:, :, :, 2] -= 123.68
    return img


class TrainingSet:
    """
    Class for getting generators to iterate over training examples.
    These are located at a cloud storage service and are loaded dynamically.
    """
    
    def __init__(self):
        """Ask for username/password for PCloud access."""
        print("Logging into pCloud account.")
        username = input("Username: ")
        password = getpass()
        self.cloud = PCloud(username, password)
        print("Success!")
        # These are initialized with the corresponding method:
        self.training = None
        self.validation = None

    def initialize(self, input_folder, target_folder,
                   validation_folder=None, validation_target_folder=None,
                   batch_divider=10):
        """
        Initialize generators for iterating over training examples
        and if specified also get generator for validation examples.
        Arguments take the path/to/folder in cloud storage.
        """
        x_files, y_files = self._filelist(input_folder, target_folder)
        # HACK Set batch size to 1.
        self.training = self._create_generator(x_files, y_files,
                                               batch_divider)
        if validation_folder and validation_target_folder:
            x_vali, y_vali = self._filelist(validation_folder, 
                                            validation_target_folder)
            # HACK Set batch size to 1.
            self.validation = self._create_generator(x_vali, y_vali,
                                                     batch_divider)
            
    def _filelist(self, input_folder, target_folder):
        """
        Get a 2-entry list of the files from the cloud storage.
        Important: The inner lists contain tuples (filename, fileid).
        """
        files = [self.cloud.get_files_in_folder(*folder.split("/"))
                 for folder in (input_folder, target_folder)]
        self.files = files
        # Is is necessary to sort the file lists? Doesn't seem so.
        assert len(files[0]) == len(files[1])
        # Check if you have fitting input and target file pair.
        checkpairs = map(lambda inpt, trgt: inpt[0][:-5] == trgt[0][:-4],
                         files[0], files[1])
        for check in checkpairs:
            assert check
        return files
        
    def _create_generator(self, x_files, y_files,
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
        i = 0
        while True:
            inputs = self._file_to_images(x_files[i])
            targets = self._file_to_array(y_files[i])
            if batch_div:
                for j in range(0, img_per_file, batch_size):
                    yield (inputs[j:j+batch_size],
                           targets[j:j+batch_size])
            else:
                yield (inputs, targets)
            i += 1
            # If the end of the list is reached start again.
            if i == len(x_files):
                i = 0
    
    def _find_divider(self, divider, divisor):
        for i in range(divider, 0, -1):
            if divisor % divider == 0:
                return divider
            else:
                divider += 1
                
    def _file_to_images(self, file, xtiles=10, ytiles=10):
        raw = self.cloud.get_file(file[1])
        raw_array = np.frombuffer(raw.read(), dtype=np.int8)
        montage = cv2.imdecode(raw_array, cv2.IMREAD_COLOR)
        ysize = montage.shape[0]
        xsize = montage.shape[1]
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
        
    def _file_to_array(self, file, lines=100):
        mapping = {"dres":0, "japa":1, "nude":2, "scho":3, "shir":4, "swim":5}
        arrays = np.empty((lines, len(mapping)), dtype=np.bool)
        raw = self.cloud.get_file(file[1])
        i = 0
        for line in raw:
            line = line.decode()[2:6]
            array = np.zeros(len(mapping), dtype=np.bool)
            array[mapping[line]] = True
            arrays[i] = array
            i += 1
        assert i == lines
        return arrays