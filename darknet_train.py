# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

#from yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', '', 'Input image')
tf.app.flags.DEFINE_integer('num_classes', 1000, 'number of classes')
tf.app.flags.DEFINE_integer('y', 0, 'image class')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')


def dark_net(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    with tf.variable_scope('darknet-53'):
        y_hat = darknet53(inputs)

    return y_hat

def one_hot(ys, num_classes):
    vec = np.zeros((len(ys),num_classes))
    for i,y in enumerate(ys):
        vec[i][y]=1

def main(argv=None):
    img = Image.open(FLAGS.input_img)
    img_resized = img.resize(size=(FLAGS.size, FLAGS.size))

    y = FLAGS.y

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

    with tf.Session() as sess:
        sess.run(load_ops)

        y_hat = sess.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})


if __name__ == '__main__':
    tf.app.run()
