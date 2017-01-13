#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 07:52:00 2017

@author: Thomas Leyh
"""

import cv2
import numpy as np
import threading
from getpass import getpass
from collections import deque
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
        self.training = self._create_generator(x_files, y_files,
                                               batch_divider)
        if validation_folder and validation_target_folder:
            x_vali, y_vali = self._filelist(validation_folder, 
                                            validation_target_folder)
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
        image_generator = self._retrieve_raw_data(x_files)
        target_generator = self._retrieve_raw_data(y_files)
        while True:
            inputs = self._raw_to_images(next(image_generator))
            targets = self._raw_to_array(next(target_generator))
            if batch_div:
                for j in range(0, img_per_file, batch_size):
                    yield (inputs[j:j+batch_size],
                           targets[j:j+batch_size])
            else:
                yield inputs, targets
    
    def _find_divider(self, divider, divisor):
        for i in range(divider, 0, -1):
            if divisor % divider == 0:
                return divider
            else:
                divider += 1
    
    def _retrieve_raw_data(self, files,
                           initial_size=4,
                           lower_limit=3,
                           step=3):
        # Initialize queue and append values the first time.
        queue = deque()
        temp = files.copy()
        line = 0
        # Initialize query with the first few values.
        for i in range(line, initial_size):
            raw = self.cloud.get_file(files[i][1])
            queue.append(raw)
        line += initial_size
        lock = False
        while True:
            yield queue.popleft()
            # If the queue is getting too small, download more data and fill it in.
            if len(queue) < lower_limit:
                # Start a background process to download the data.
                if not lock:
                    raws = []
                    p = threading.Thread(
                                    target=self._get_raw_from_cloud,
                                    args=(files[line:line + step], raws))
                    p.start()
                    lock = True
                # If process finished append its downloaded data to queue.
                if not p.is_alive() or len(queue) == 0:
                    p.join()
                    queue.extend(raws)
                    lock = False
                    line += step
                    # This line is necessary to loop indefinitely over the data.
                    if line + step > len(files):
                        files = files[line:]
                        files.extend(temp)
                        line = 0
    
    def _get_raw_from_cloud(self, files, output):
        for file in files:
            output.append(self.cloud.get_file(file[1]))
        
    def _raw_to_images(self, raw, xtiles=10, ytiles=10):
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
        
    def _raw_to_array(self, raw, lines=100):
        mapping = {"dres":0, "japa":1, "nude":2, "scho":3, "shir":4, "swim":5}
        arrays = np.empty((lines, len(mapping)), dtype=np.bool)
        i = 0
        for line in raw:
            line = line.decode()[2:6]
            array = np.zeros(len(mapping), dtype=np.bool)
            array[mapping[line]] = True
            arrays[i] = array
            i += 1
        assert i == lines
        return arrays
