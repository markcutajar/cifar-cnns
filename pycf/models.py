import tensorflow as tf
from tensorflow.contrib import layers as cls


def fc6l(learning_rate=0.001, optimizer=tf.train.AdadeltaOptimizer):

    # Define training optimizer with specified learning rate
    train_optimizer = optimizer(learning_rate=learning_rate)

    # Setup different layers for the network
    with tf.name_scope('data'):
        inputs = tf.placeholder(tf.float32, shape=[None, 3072])
        targets = tf.placeholder(tf.float32, shape=[None, 10])

    out_fl1 = cls.fully_connected(inputs, 3000, normalizer_fn=cls.batch_norm, scope='fcl-1')
    out_fl2 = cls.fully_connected(out_fl1, 3000, normalizer_fn=cls.batch_norm, scope='fcl-2')
    out_fl3 = cls.fully_connected(out_fl2, 1000, normalizer_fn=cls.batch_norm, scope='fcl-3')
    out_fl4 = cls.fully_connected(out_fl3, 1000, normalizer_fn=cls.batch_norm, scope='fcl-4')
    out_fl5 = cls.fully_connected(out_fl4, 300, normalizer_fn=cls.batch_norm, scope='fcl-5')
    output = cls.fully_connected(out_fl5, 10, activation_fn=None, scope='output')

    # Setup metric and training operations
    accuracy, error, train_op = setup_metrics(output, targets, train_optimizer)
    summary_op = summary(accuracy=accuracy, error=error)

    # Package as seperate dictionaries and return
    return package_return(accuracy, error, summary_op, train_op, inputs, targets)


"""
def two_conv_two_fc_model(learning_rate=0.001):

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

    with tf.name_scope('data'):
        inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        targets = tf.placeholder(tf.float32, shape=[None, 10])

    with tf.variable_scope('conv2-1'):
        out_cv1 = cls.conv2d(inputs, 5, 6)

    with tf.variable_scope('conv2-2'):
        out_cv2 = cls.conv2d(out_cv1, 5, 8)

    with tf.variable_scope('fcl-3'):
        out_cv2_flattened = cls.flatten(out_cv2)
        out_fl1 = cls.fully_connected(out_cv2_flattened, 50)

    with tf.variable_scope('fcl-4'):
        output = cls.fully_connected(out_fl1, 10)

    with tf.name_scope('error'):
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=targets))

    with tf.name_scope('train'):
        train_op = optimizer.minimize(error)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(output, 1), tf.argmax(targets, 1)), tf.float32))

    tf.summary.scalar('error', error)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    return package_return(accuracy, error, summary_op, train_op, inputs, targets)
"""


def setup_metrics(predictions, labels, optimizer):

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1)), tf.float32))

    with tf.name_scope('error'):
        error = tf.losses.softmax_cross_entropy(logits=predictions, onehot_labels=labels,
                                                reduction=tf.losses.Reduction.MEAN)
    with tf.name_scope('train'):
        train_op = optimizer.minimize(error)

    return accuracy, error, train_op




def accuracy_op(predictions, labels):
    return


def summary(accuracy, error):
    tf.summary.scalar('error', error)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()
    return summary_op


def package_return(accuracy, error, summary_op, train_op, inputs, targets):
    metrics = {
        'accuracy': accuracy,
        'error': error
    }
    ops = {
        'summary': summary_op,
        'train': train_op
    }
    data = {
        'inputs': inputs,
        'targets': targets
    }
    return metrics, ops, data
