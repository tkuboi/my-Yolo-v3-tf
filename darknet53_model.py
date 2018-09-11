# ==============================================================================
# 
# ==============================================================================
"""Contains definitions for Darknet53 Networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re 
import glob
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from random import shuffle

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES
BATCH_SIZE = 100
LIMIT = 1000
IMAGE_SIZE = 256
NUM_EPOCH = 10

def batch_norm(inputs, training, data_format):
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)

def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                               [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    if strides > 1: 
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)

def _darknet53_block(inputs, filters, strides, data_format):
    shortcut = inputs
    inputs = conv2d_fixed_padding(inputs, filters, 1, data_format, strides)
    inputs = conv2d_fixed_padding(inputs, filters * 2, 3, data_format, strides)
    return inputs + shortcut

def darknet53_base(inputs, data_format):
    inputs = conv2d_fixed_padding(inputs, 32, 3, data_format, strides=1)
    inputs = conv2d_fixed_padding(inputs, 64, 3, data_format, strides=2)
    inputs = _darknet53_block(inputs, 32, 1, data_format)
    inputs = conv2d_fixed_padding(inputs, 128, 3, data_format, strides=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64, 1, data_format)

    inputs = conv2d_fixed_padding(inputs, 256, 3, data_format, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128, 1, data_format)

    inputs = conv2d_fixed_padding(inputs, 512, 3, data_format, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 256, 1, data_format)

    inputs = conv2d_fixed_padding(inputs, 1024, 3, data_format, strides=2)

    for i in range(4):
        inputs = _darknet53_block(inputs, 512, 1, data_format)

    return inputs

def darknet53(inputs, data_format):
    # transpose the inputs to NCHW
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    inputs = darknet53_base(inputs, data_format)
    #shape = tf.shape(inputs)
    shape=inputs.get_shape().as_list()
    #inputs = tf.layers.average_pooling2d(inputs, [8,1], [1,1],
    inputs = tf.layers.average_pooling2d(inputs, shape[2:], [1,1],
                                         padding='valid',
                                         data_format=data_format)
    inputs = tf.layers.dense(inputs, 1000, 'softmax')
    #inputs = tf.nn.softmax(inputs, axis=None)
    return inputs

def one_hot(label, num_classes):
    vec = np.zeros(num_classes)
    digit = re.find(r'\d+', label)
    c = int(digit) - BASE 
    vec[c]=1
    return vec

def get_image_list(directory):
    images = []
    classes = []
    for subdir in next(os.walk(direcotry))[1]:
        images += glob.glob(directory + '/' + subdir + '/*.jpeg')
        classes += [subdir] * len(images)
    return zip(images, classes)

def get_images_labels(images_labels):
    x = []
    y = []
    for image,label in image_labels:
        img = Image.open(filename)
        x.append(img.resize(size=(IMAGE_SIZE, IMAGE_SIZE)))
        y.append(one_hot(label, 1000))
    return np.array(x), np.array(y)

# tf Graph Input
inputs = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.float32, [None, 1000]) # 0-9 digits recognition => 10 classes

y_hat = darknet53(inputs, 'channels_first') 

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

images_labels = get_image_list(directory)
images_labels = shuffle(images_labels)
images_labels, rest = images_labels[:LIMIT], images_labels[LIMIT:]

total_batch = len(image_labels) / BATCH_SIZE

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     for e in range(NUM_EPOCH):
         for i in range(total_batch):
             batch_x_y = image_labels[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
             images,labels = get_images_labels(batch_x_y) 
             _, loss = sess.run([optimizer, cost], feed_dict={inputs: images, y: labels})             print loss 
