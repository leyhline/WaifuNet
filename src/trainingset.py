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
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from getpass import getpass
from collections import deque
from .pcloud import PCloud


logging.basicConfig(filename="train.log", filemode="w", level=logging.INFO,
                    format = "%(asctime)s - %(levelname)s - %(message)s")


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
            assert check, "Training and target data does not fit together."
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
            image_name, image_raw = next(image_generator)
            target_name, target_raw = next(target_generator)
            # Check if you really got the right image-target combo.
            logging.info("Files {} and {} received.".format(image_name, target_name))
            assert image_name[:-5] == target_name[:-4], "{} does not fit to {}.".format(image_name, target_name)
            inputs = self._raw_to_images(image_raw)
            targets = self._raw_to_array(target_raw)
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
    
    def _retrieve_raw_data(self, files,
                           initial_size=64,
                           lower_limit=32,
                           step=64):
        """
        Some kind of data structure where the necessary data is buffered and
        loaded in advance per simple multithreading.
        Loops indefinetly over data via generator.
        """
        # Initialize queue and append values the first time.
        queue = deque()
        temp = files.copy()
        line = 0
        # Initialize query with the first few values.
        raws = []
        self._get_raw_from_cloud(files[line:initial_size], raws)
        queue.extend(raws)
        line += initial_size
        lock = False
        while True:
            yield queue.popleft()
            # If the queue is getting too small, download more data and fill it in.
            if len(queue) < lower_limit:
                # Start a background thread to download the data.
                if not lock:
                    raws = []
                    p = threading.Thread(
                                    target=self._get_raw_from_cloud,
                                    args=(files[line:line + step], raws))
                    p.start()
                    lock = True
                # If thread finished append its downloaded data to queue.
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
    
    def _get_raw_from_cloud(self, files, output, workers=4):
        """Helper function for getting data via seperate thread."""
        name, fileid = zip(*files)
        with ThreadPoolExecutor(max_workers=workers) as e:
            raw = e.map(self.cloud.get_file, fileid)
        output.extend(zip(name, raw))
        
    def _raw_to_images(self, raw, xtiles=10, ytiles=10):
        """Take raw data (actually it's a class from requests package)
           and decode it to a batch of image arrays. (4 dimensions)"""
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
        """Take raw data (actually it's a class from requests package)
           and decode it to a batch of binary arrays for classification. (4 dimensions)"""
        mapping = {"dres":0, "japa":1, "nude":2, "scho":3, "shir":4, "swim":5}
        arrays = np.empty((lines, len(mapping)), dtype=np.bool)
        i = 0
        for line in raw:
            line = line.decode()[2:6]
            array = np.zeros(len(mapping), dtype=np.bool)
            array[mapping[line]] = True
            arrays[i] = array
            i += 1
        # Assert correct batch size.
        assert i == lines
        return arrays
