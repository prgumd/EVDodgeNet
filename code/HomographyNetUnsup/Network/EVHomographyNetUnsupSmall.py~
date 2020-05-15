import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def EVHomographyNetUnsup(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    # Img is of size MxNx3

    # conv1 output is of size M/2 x N/2 x 64
    conv1 = tf.layers.conv2d(inputs=Img, filters=64, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv1T')

    # bn1 output is of size M/2 x N/2 x 64
    bn1 = tf.layers.batch_normalization(conv1, name='bn1T')

    # bn1 output is of size M/2 x N/2 x 64
    bn1 = tf.nn.relu(bn1, name='relu1T')

    # conv2 output is of size M/4 x N/4 x 64
    conv2 = tf.layers.conv2d(inputs=bn1, filters=64, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv2T')

    # bn2 output is of size M/4 x N/4 x 64
    bn2 = tf.layers.batch_normalization(conv2, name='bn2T')

    # bn2 output is of size M/4 x N/4 x 64
    bn2 = tf.nn.relu(bn2, name='relu2T')

    # conv3 output is of size M/8 x N/8 x 128
    conv3 = tf.layers.conv2d(inputs=bn2, filters=128, kernel_size=[7, 7], strides=(2, 2), padding="same", activation=None, name='conv3T')

    # bn3 output is of size M/8 x N/8 x 128
    bn3 = tf.layers.batch_normalization(conv3, name='bn3T')

    # bn3 output is of size M/8 x N/8 x 128
    bn3 = tf.nn.relu(bn3, name='relu3T')

    # conv4 output is of size M/16 x N/16 x 128
    conv4 = tf.layers.conv2d(inputs=bn3, filters=128, kernel_size=[7, 7], strides=(2, 2), padding="same", activation=None, name='conv4T')

    # bn4 output is of size M/16 x N/16 x 128
    bn4 = tf.layers.batch_normalization(conv4, name='bn4T')

    # bn4 output is of size M/16 x N/16 x 128
    bn4 = tf.nn.relu(bn4, name='relu4T')

    # flat is of size 1 x M/16*N/16*128
    flat = tf.reshape(bn4, [-1, ImageSize[0]*ImageSize[1]*128/(16*16)], name='flat')

    # flatdrop is a dropout layer
    flatdrop = tf.layers.dropout(flat, rate=0.75)

    # fc1 
    fc1 = tf.layers.dense(flatdrop, units=1024, activation=None)

    # fc2
    fc2 = tf.layers.dense(fc1, units=8, activation=None)

    return fc2

