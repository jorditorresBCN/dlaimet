import sys
import tensorflow as tf
import datetime
import cifar10

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope

IMG_SIZE = 224
LEARNING_RATE = 0.0001
WD = 1e-6
TRAINING_SET = 50000
N_EPOCHS = 10
BATCH_SIZE = 32
DATASET_DIR = "cifar10_data"
N_ITERATIONS = int(TRAINING_SET / BATCH_SIZE)

tensorboard_dir = "tb"
tensorboard_active = False


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19'):
    """Oxford Net VGG 19-Layers version E Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with variable_scope.variable_scope(scope, 'vgg_19', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with arg_scope(
                [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
                outputs_collections=end_points_collection):
            net = layers_lib.repeat(
                inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
            net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
            net = layers_lib.repeat(net, 4, layers.conv2d, 256, [3, 3], scope='conv3')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
            net = layers_lib.repeat(net, 4, layers.conv2d, 512, [3, 3], scope='conv4')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
            net = layers_lib.repeat(net, 4, layers.conv2d, 512, [3, 3], scope='conv5')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            net = layers_lib.dropout(
                net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
            net = layers_lib.dropout(
                net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            net = layers.conv2d(
                net,
                num_classes, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope='fc8')
            # Convert end_points_collection into a end_point dict.
            end_points = utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points


def train():
    x, labels = cifar10.get_inputs(False, BATCH_SIZE)
    x2 = tf.image.resize_images(x, (IMG_SIZE, IMG_SIZE))
    logits, end_points = vgg_19(x2, 10, is_training=True)
    labels = tf.one_hot(labels, depth=10)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=WD).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    sess = tf.InteractiveSession()
    if tensorboard_active:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coordinator = tf.train.Coordinator()
    th = tf.train.start_queue_runners(coord=coordinator)

    time = None
    # Train
    for epochid in range(N_EPOCHS):
        print("Running epoch %d/%d ..." % ((epochid + 1), N_EPOCHS))
        for iterid in range(N_ITERATIONS):
            start = datetime.datetime.now()
            percent = (100 * (iterid + 1)) / N_ITERATIONS
            if time is None:
                sys.stdout.write(
                    '\r %.f%% (%d/%d) \r\n ETA (All training): Estimating...' % (percent, (iterid + 1), N_ITERATIONS))
            else:
                total_time = time * (N_ITERATIONS - iterid - 1) * (N_EPOCHS - epochid - 1)
                total_time_h = int(total_time / 3600)
                total_time_min = int((total_time % 3600) / 60)
                total_time_sec = int((total_time % 60))
                sys.stdout.write('\r %.f%% (%d/%d)  ETA (All training): %d h %d m %d s' % (
                    percent, (iterid + 1), N_ITERATIONS, total_time_h, total_time_min, total_time_sec))

            if tensorboard_active:
                if iterid % 10 == 0:
                    summary, acc, _ = sess.run([merged, accuracy, train_step])
                    train_writer.add_summary(summary, iterid)
                else:
                    summary, _ = sess.run([merged, train_step])
                    train_writer.add_summary(summary, iterid)
            else:
                sess.run(train_step)
            time = datetime.datetime.now() - start
            time = time.total_seconds()

            # Test trained model
            # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print(" --> Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images,
            #                                                        y_: mnist.test.labels}))
    coordinator.request_stop()
    coordinator.join(th)


def main(_):
    cifar10.maybe_download_and_extract(DATASET_DIR)
    train()


if __name__ == '__main__':
    tf.app.run(main=main)
