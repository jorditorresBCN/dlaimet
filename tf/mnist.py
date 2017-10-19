# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

TRAINING_SET = 60000

BATCH_SIZE = 64
LEARNING_RATE = 0.01
N_EPOCHS = 3
N_ITERATIONS = int(TRAINING_SET/BATCH_SIZE)

tensorboard_dir = "tb/run1"
tensorboard_active = True


def model(x):
    x2 = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image("images", x2)
    net = tf.layers.conv2d(x2, 20, [5, 5], activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2])
    net = tf.layers.conv2d(net, 50, [5, 5], activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2])
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(net, 500, activation=tf.nn.relu)
    return tf.layers.dense(net, 10)


def get_loss(y_, y):
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy


def get_train_step(cross_entropy):
    with tf.name_scope('train'):
        train_step = tf.train.AdadeltaOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
    return train_step


def get_accuracy(y_, y):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def train(mnist, sess, x, y_, accuracy, train_step, train_writer=None, test_writer=None, merged=None):
    global_step = 0
    for epochid in range(N_EPOCHS):
        print("Running epoch %d ..." % (epochid + 1))
        for iterid in range(N_ITERATIONS):
            percent = (100 * (iterid + 1)) / N_ITERATIONS
            sys.stdout.write('\r %.f%% (%d/%d) ' % (percent, (iterid + 1), N_ITERATIONS))
            batch_xs, batch_ys = mnist.train.next_batch(batch_size=BATCH_SIZE)
            if tensorboard_active:
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
                train_writer.add_summary(summary, global_step)
            else:
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            global_step += 1

        # Test trained model
        if tensorboard_active:
            acc, summary = sess.run([accuracy, merged], feed_dict={x: mnist.test.images,
                                                                   y_: mnist.test.labels})
            test_writer.add_summary(summary, global_step)
        else:
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print(" --> Accuracy: ", acc)


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y = model(x)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = get_loss(y_, y)
    train_step = get_train_step(cross_entropy)
    accuracy = get_accuracy(y_, y)

    sess = tf.InteractiveSession()

    if tensorboard_active:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(tensorboard_dir + '/train',
                                             sess.graph)
        test_writer = tf.summary.FileWriter(tensorboard_dir + '/test')
    else:
        train_writer = None
        test_writer = None
        merged = None
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    train(mnist, sess, x, y_, accuracy, train_step, train_writer, test_writer, merged)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
