import os
import datetime
import tensorflow as tf

import pycf.models as models
from pycf.data_providers import CIFAR10DataProvider

# Set log level to suppress build warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    # Create Graph
    modelA = tf.Graph()

    num_epochs = 10
    batch_size = 100
    total_batches = 400

    with modelA.as_default():
        # Create dataset and function for next example
        train_data = CIFAR10DataProvider(batch_size=batch_size, epochs=num_epochs, shape='2d')
        next_example, next_label = train_data.next()

        # Define model
        metrics_ops, train_ops = models.fc6(next_example, next_label)

    with modelA.as_default():

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_writer = tf.summary.FileWriter(os.path.join('tf-log', timestamp, 'train'), graph=tf.get_default_graph())

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            running_err = 0
            running_acc = 0
            train_step = 0

            while train_step < num_epochs * total_batches:
                train_info, metrics = sess.run([train_ops, metrics_ops])

                train_writer.add_summary(train_info['summary'], train_step)
                running_err += metrics['error']
                running_acc += metrics['accuracy']

                if (train_step % total_batches == 0) and (train_step != 0):
                    print('Batch {0:02}\t| Average Error = {1:02.2f}\t| Average Accuracy = {2:.4f}'
                          .format(train_step, running_err / total_batches, running_acc / total_batches))

                    running_err = 0
                    running_acc = 0
                train_step += 1
