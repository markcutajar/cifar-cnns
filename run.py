import os
import datetime
import numpy as np
import tensorflow as tf

from pycf.models import two_layer_fc_model
from pycf.models import four_layer_fc_model
from pycf.data_providers import CIFAR10DataProvider

if __name__ == '__main__':
    # Set log level to remove build warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load Training data
    train_data = CIFAR10DataProvider(shape='2d')

    # Create graph
    modelA = tf.Graph()

    # Define model
    with modelA.as_default():
        modelInfo = four_layer_fc_model()

        # Set logging files for tensorboard
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_writer = tf.summary.FileWriter(os.path.join('tf-log', timestamp, 'train'), graph=modelA)

    # Train model
    with modelA.as_default():
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            num_epoch = 20
            for epochs in range(num_epoch):
                running_error = 0
                for batch_num, (input_batch, target_batch) in enumerate(train_data):
                    _, batch_error, summary = sess.run([modelInfo.get('train_op'),
                                                        modelInfo.get('error'),
                                                        modelInfo.get('summary_op')],

                                                       feed_dict={modelInfo.get('inputs'): input_batch,
                                                                  modelInfo.get('targets'): target_batch})

                    running_error += batch_error
                    train_writer.add_summary(summary, epochs * train_data.num_batches + batch_num)

                running_error /= train_data.num_batches
                print('End of epoch {0}: Average epoch error = {1:.2f}'.format(epochs + 1, running_error))
