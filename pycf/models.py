import tensorflow as tf


def two_layer_fc_model(learning_rate=0.001):

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

    with tf.name_scope('data'):
        inputs = tf.placeholder(tf.float32, shape=[None, 3072])
        targets = tf.placeholder(tf.float32, shape=[None, 10])

    with tf.variable_scope('fcl-1'):
        out_fl1 = tf.contrib.layers.fully_connected(inputs, 1000)

    with tf.variable_scope('fcl-2'):
        out_fl2 = tf.contrib.layers.fully_connected(out_fl1, 10)

    with tf.name_scope('error'):
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_fl2, labels=targets))

    with tf.name_scope('train'):
        train_op = optimizer.minimize(error)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(out_fl2, 1), tf.argmax(targets, 1)), tf.float32))

    tf.summary.scalar('error', error)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    return {'accuracy': accuracy,
            'error': error,
            'summary_op': summary_op,
            'train_op': train_op,
            'inputs': inputs,
            'targets': targets}


def four_layer_fc_model(learning_rate=0.001):

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

    with tf.name_scope('data'):
        inputs = tf.placeholder(tf.float32, shape=[None, 3072])
        targets = tf.placeholder(tf.float32, shape=[None, 10])

    with tf.variable_scope('fcl-1'):
        out_fl1 = tf.contrib.layers.fully_connected(inputs, 1000)

    with tf.variable_scope('fcl-2'):
        out_fl2 = tf.contrib.layers.fully_connected(out_fl1, 1000)

    with tf.variable_scope('fcl-3'):
        out_fl3 = tf.contrib.layers.fully_connected(out_fl2, 1000)

    with tf.variable_scope('fcl-4'):
        output = tf.contrib.layers.fully_connected(out_fl3, 10)

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

    return {'accuracy': accuracy,
            'error': error,
            'summary_op': summary_op,
            'train_op': train_op,
            'inputs': inputs,
            'targets': targets}


def two_conv_two_fc_model(learning_rate=0.001):

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

    with tf.name_scope('data'):
        inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        targets = tf.placeholder(tf.float32, shape=[None, 10])

    with tf.variable_scope('conv2-1'):
        out_cv1 = tf.contrib.layers.conv2d(inputs, 5, 6)

    with tf.variable_scope('conv2-2'):
        out_cv2 = tf.contrib.layers.conv2d(out_cv1, 5, 8)

    with tf.variable_scope('fcl-3'):
        out_cv2_flattened = tf.contrib.layers.flatten(out_cv2)
        out_fl1 = tf.contrib.layers.fully_connected(out_cv2_flattened, 50)

    with tf.variable_scope('fcl-4'):
        output = tf.contrib.layers.fully_connected(out_fl1, 10)

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

    return {'accuracy': accuracy,
            'error': error,
            'summary_op': summary_op,
            'train_op': train_op,
            'inputs': inputs,
            'targets': targets}