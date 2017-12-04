# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import tensorflow as tf
import json
import os
from config import settings
from pycf import data_prep

from tensorflow.python.lib.io import file_io


class CIFAR10DataProvider(object):
    """Data provider for CIFAR-10 object images."""

    def __init__(self, which_set='train', batch_size=100,
                 shuffle_order=True, epochs=10, shape='3d'):
        """Create a new CIFAR-10 data provider object.

        :arg: which_set: One of 'train', 'valid' or 'test'. Determines which
              portion of the CIFAR-10 data this object should provide.
        :arg: batch_size (int): Number of data points to include in each batch.
        :arg: max_num_batches (int): Maximum number of batches to iterate over
              in an epoch. If `max_num_batches * batch_size > num_data` then
              only as many batches as the data can be split into will be
              used. If set to -1 all of the data will be used.
        :arg: shuffle_order (bool): Whether to randomly permute the order of
              the data before each epoch.
        :arg: rng (RandomState): A seeded random number generator.
        """

        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train or valid. '
            'Got {0}'.format(which_set))
        self.which_set = which_set
        if shape not in ('2d', '3d'):
            raise ValueError('Shape can only be \'2d\' or \'3d\'. Given {}'.format(shape))
        self.shape = shape

        # Set batches and total num
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size

        if self.which_set == 'train':
            self._total_batches = int(40000 / batch_size)
        else:
            self._total_batches = int(10000 / batch_size)

        # Load data
        self._load_metadata()
        self._load_dataset()

        # Shuffle data
        if shuffle_order:
            self.dataset = self.dataset.shuffle(buffer_size=10000, seed=settings.SEED)

        # Update batches and set iterators
        self._update_batch()
        self.set_epochs(epochs)
        self.iterator = self.dataset.make_one_shot_iterator()

    def _load_metadata(self):
        """Load the metadata and confirm files exist."""
        # If TFRecords do not exist prep data
        if (not os.path.exists(settings.TFR_META)
                or not os.path.exists(settings.TFR_TRAIN)
                or not os.path.exists(settings.TFR_VALID)
                or not os.path.exists(settings.TFR_TEST)):
            data_prep.run()

        # Load metadata and save variables
        with file_io.FileIO(settings.TFR_META, 'r') as f:
            metadata = json.load(f)
        self.num_classes = metadata['num_classes']
        self.label_map = metadata['label_map']
        self._height = metadata['height']
        self._depth = metadata['depth']
        self._width = metadata['width']

    def _load_dataset(self):
        """Create dataset from the TFRecords data files."""
        self.files = {'train': settings.TFR_TRAIN, 'valid': settings.TFR_VALID, 'test': settings.TFR_TEST}
        self.dataset = tf.data.TFRecordDataset(self.files[self.which_set])

        # Parse data from strings and set
        self.dataset = self.dataset.map(self._parse_function)

    def _parse_function(self, example_proto):
        """Convert from strings to higher dim tensors."""
        features = {"image_raw": tf.FixedLenFeature((), tf.string),
                    "label_raw": tf.FixedLenFeature((), tf.string)}

        parsed_features = tf.parse_single_example(example_proto, features)

        # Decode labels and set to one hot encoding
        label = tf.decode_raw(parsed_features["label_raw"], tf.int32)[0]
        one_of_k_targets = tf.one_hot(label, self.num_classes)

        # Decode image
        image = tf.cast(tf.decode_raw(parsed_features["image_raw"], tf.int8), tf.float32)

        # If image 3d reshape
        if self.shape is '3d':
            image = tf.split(image, self._depth)
            image = tf.stack(image, axis=-1)
            image = tf.reshape(image, [self._height, self._width, self._depth])
        else:
            image = tf.reshape(image, [self._height * self._width * self._depth])

        return image, one_of_k_targets

    def get_dataset(self):
        """Return the whole dataset."""
        return self.dataset

    def label_map(self):
        """Return label map"""
        return self.label_map

    def iterator(self):
        """Get iterator."""
        return self.iterator

    def next(self):
        """Get operation for next item."""
        return self.iterator.get_next()

    def get_shape(self):
        return self.dataset.output_shapes

    def set_epochs(self, num_epochs):
        """Set the number of epochs to run."""
        if num_epochs < 1:
            raise ValueError('num_epochs must be >= 1')
        self.dataset = self.dataset.repeat(num_epochs)

    @property
    def batch_size(self):
        """Get the number of data points in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Set the number of data points in each batch."""
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_batch()

    def _update_batch(self):
        """Update the batches of the dataset."""
        self.dataset = self.dataset.batch(self._batch_size)

    def total_batches(self):
        """Get the total number of available batches"""
        return self._total_batches
