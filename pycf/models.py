import tensorflow as tf
from tensorflow.contrib import layers as cls


def fc6(inputs, targets, learning_rate=0.001, optimizer=tf.train.AdadeltaOptimizer):

    # Define training optimizer with specified learning rate
    train_optimizer = optimizer(learning_rate=learning_rate)

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
    return package_return(accuracy, error, summary_op, train_op)


def cv3fc2(inputs, targets,
           width=(5, 5, 5), depth=(6, 8, 8),
           learning_rate=0.001, optimizer=tf.train.AdadeltaOptimizer):

    fc_width = round((inputs.shape[0] * inputs.shape[1] * depth[-1]) / 4)

    # Define training optimizer with specified learning rate
    train_optimizer = optimizer(learning_rate=learning_rate)

    out_cv1 = cls.conv2d(inputs, width[0], depth[0], normalizer_fn=cls.batch_norm, scope='cv-1')
    out_cv2 = cls.conv2d(out_cv1, width[1], depth[1], normalizer_fn=cls.batch_norm, scope='cv-2')
    out_cv3 = cls.conv2d(out_cv2, width[2], depth[2], normalizer_fn=cls.batch_norm, scope='cv-3')
    out_cv3_flattened = cls.flatten(out_cv3, scope='flt')
    out_fl1 = cls.fully_connected(out_cv3_flattened, fc_width, normalizer_fn=cls.batch_norm, scope='fcl-1')
    out_fl2 = cls.fully_connected(out_fl1, round(fc_width/4), normalizer_fn=cls.batch_norm, scope='fcl-2')
    output = cls.fully_connected(out_fl2, 10, activation_fn=None, scope='output')

    # Setup metric and training operations
    accuracy, error, train_op = setup_metrics(output, targets, train_optimizer)
    summary_op = summary(accuracy=accuracy, error=error)

    # Package as seperate dictionaries and return
    return package_return(accuracy, error, summary_op, train_op)


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


def package_return(accuracy, error, summary_op, train_op):
    metrics = {
        'accuracy': accuracy,
        'error': error
    }
    ops = {
        'summary': summary_op,
        'train': train_op
    }
    return metrics, ops
