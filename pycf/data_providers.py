# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.

Credit goes to the University of Edinburgh Team. Namely the team handling the machine learning practical repository at:
https://github.com/CSTR-Edinburgh/mlpractical
"""

import os
import pickle
import numpy as np
from config import settings


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        :arg: inputs (ndarray): Array of data input features of shape (num_data, input_dim).
        :arg: targets (ndarray): Array of data output targets of shape
              (num_data, output_dim) or (num_data,) if output_dim == 1.
        :arg: batch_size (int): Number of data points to include in each batch.
        :arg: max_num_batches (int): Maximum number of batches to iterate over
              in an epoch. If `max_num_batches * batch_size > num_data` then
              only as many batches as the data can be split into will be
              used. If set to -1 all of the data will be used.
        :arg: shuffle_order (bool): Whether to randomly permute the order of
              the data before each epoch.
        :arg: rng (RandomState): A seeded random number generator.
        """

        self.inputs = inputs
        self.targets = targets

        # Select batch size
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size

        # Set maximum number of batches used
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches

        # Update batches iteration scheme
        self._update_num_batches()

        # Set shuffle items
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(settings.SEED)
        self.rng = rng

        self._curr_batch = 0
        self.new_epoch()

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
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Get the maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        """Set the maximum number of batches to iterate over in an epoch."""
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface."""
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch

    def __next__(self):
        return self.next()


class OneOfKDataProvider(DataProvider):
    """1-of-K classification target data provider.

    Transforms integer target labels to binary 1-of-K encoded targets.

    Derived classes must set self.num_classes appropriately.
    """

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(OneOfKDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1-of-K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1-of-K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        return one_of_k_targets


class CIFAR10DataProvider(OneOfKDataProvider):
    """Data provider for CIFAR-10 object images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, shape='3d'):
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

        if shape not in ('2d', '3d'):
            raise ValueError('Shape can only be \'2d\' or \'3d\'. Given {}'.format(shape))

        num_rgb_layers = 3

        image_data = []
        image_labels = []
        self.label_map = []
        self.filenames = []
        self.which_set = which_set

        self.batch_files = {'train': settings.TRAIN_FILES.split(','), 'valid': settings.VALID_FILES.split(','),
                            'test': settings.TEST_FILES.split(',')}

        # Get data and labels
        for filename in self.batch_files[self.which_set]:
            with open(filename.strip(), 'rb') as file:
                loaded_data = pickle.load(file, encoding='bytes')
            image_data.extend(loaded_data.get(b'data'))
            image_labels.extend(loaded_data.get(b'labels'))
            self.filenames.extend(loaded_data.get(b'filenames'))

        self.data = np.asarray(image_data)
        self.labels = np.asarray(image_labels)

        if image_data is None or self.labels is None or self.filenames is None:
            raise ValueError('File(s) not loaded correctly: {}'.format(', '.join(self.batch_files)))

        # Load metadata
        with open(settings.METADATA, 'rb') as file:
            metadata = pickle.load(file, encoding='bytes')

        # Get label map from metadata
        for item in metadata.get(b'label_names'):
            self.label_map.extend(str(item, encoding='UTF-8'))

        # Get metadata information
        self.num_classes = len(metadata.get(b'label_names'))
        image_sides = int(np.sqrt(metadata.get(b'num_vis') / num_rgb_layers))

        # Convert shape
        if shape == '3d':
            all_flat = np.swapaxes(np.reshape(self.data, (self.data.shape[0], num_rgb_layers, -1)), axis1=1, axis2=2)
            self.data = np.reshape(all_flat, (all_flat.shape[0], image_sides, image_sides, num_rgb_layers))

        # pass the loaded data to the parent class __init__
        super(CIFAR10DataProvider, self).__init__(
            self.data, self.labels, batch_size, max_num_batches, shuffle_order, rng)
