import os
import pickle
import logging
import numpy as np
import tensorflow as tf

from config import settings

# Define logger, formatter and handler
LOGGER_FORMAT = '%(levelname)s | %(message)s'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(LOGGER_FORMAT)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def run():
    batch_files = {'train': settings.TRAIN_FILES.split(','), 'valid': settings.VALID_FILES.split(','),
                   'test': settings.TEST_FILES.split(',')}

    # Iterate over all the sets to save as TFRecords
    for which_set in ('train', 'valid', 'test'):

        # Set the TFRecord filename and write
        logger.info('Saving in tfrecord set: {}'.format(which_set))
        tfrecords_filename = '{}{}_{}.tfrecords'.format(settings.DATA_LOCATION, 'cifar10', which_set)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

        # Get data and labels
        for filename in batch_files[which_set]:
            filename = filename.strip()

            if not os.path.exists(filename):
                raise ValueError('File {} does not exist'.format(filename))

            # Load file
            with open(filename, 'rb') as file:
                loaded_data = pickle.load(file, encoding='bytes')

            image_batch = loaded_data.get(b'data')
            label_batch = loaded_data.get(b'labels')
            filename_batch = loaded_data.get(b'filenames')

            # Iterate over all the samples in the file and save in TFRecord
            for img, label, path in zip(image_batch, label_batch, filename_batch):

                img_raw = np.array(img).tostring()
                label_raw = np.array(label).tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(img_raw),
                    'label_raw': _bytes_feature(label_raw),
                    'filename': _bytes_feature(path)
                }))

                writer.write(example.SerializeToString())
        writer.close()
