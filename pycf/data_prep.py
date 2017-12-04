import os
import json
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


def save_metadata():

    meta_load_location = settings.RAW_META
    meta_save_location = '{}'.format(settings.TFR_META)
    save_data = {}
    label_map = []

    # Load metadata
    with open(meta_load_location, 'rb') as file:
        metadata = pickle.load(file, encoding='bytes')

    # Get label map from metadata
    for item in metadata.get(b'label_names'):
        label_map.extend([str(item, encoding='UTF-8')])

    # Set metadata information
    save_data['num_classes'] = len(label_map)
    save_data['label_map'] = label_map
    save_data['height'] = 32
    save_data['width'] = 32
    save_data['depth'] = 3

    # Save in json file
    with open(meta_save_location, 'w') as file:
        json.dump(save_data, file)


def run():

    batch_files = {'train': settings.RAW_TRAIN.split(','), 'valid': settings.RAW_VALID.split(','),
                   'test': settings.RAW_TEST.split(',')}

    save_files = {'train': settings.TFR_TRAIN, 'valid': settings.TFR_VALID, 'test': settings.TFR_TEST}

    # Save metadata
    logger.info('Saving metadata')
    save_metadata()

    # Iterate over all the sets to save as TFRecords
    for which_set in ('train', 'valid', 'test'):

        # Set the TFRecord filename and writer
        logger.info('Saving in tfrecord set: {}'.format(which_set))
        tfrecords_filename = '{}'.format(save_files[which_set])
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


if __name__ == '__main__':
    run()
