import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def EVHomographyNetUnsupSmall(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    # Img is of size MxNx3

    # conv1 output is of size M/2 x N/2 x 16
    conv1 = tf.layers.conv2d(inputs=Img, filters=16, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv1H')

    # bn1 output is of size M/2 x N/2 x 16
    bn1 = tf.layers.batch_normalization(conv1, name='bn1H')

    # bn1 output is of size M/2 x N/2 x 16
    bn1 = tf.nn.relu(bn1, name='relu1H')

    # conv2 output is of size M/2 x N/2 x 32
    conv2 = tf.layers.conv2d(inputs=bn1, filters=32, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=None, name='conv2H')

    # bn2 output is of size M/2 x N/2 x 32
    bn2 = tf.layers.batch_normalization(conv2, name='bn2H')

    # bn2 output is of size M/2 x N/2 x 32
    bn2 = tf.nn.relu(bn2, name='relu2H')

    # conv3 output is of size M/4 x N/4 x 64
    conv3 = tf.layers.conv2d(inputs=bn2, filters=64, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv3H')

    # bn3 output is of size M/4 x N/4 x 64
    bn3 = tf.layers.batch_normalization(conv3, name='bn3H')

    # bn3 output is of size M/4 x N/4 x 64
    bn3 = tf.nn.relu(bn3, name='relu3H')

    # conv4 output is of size M/4 x N/4 x 64
    conv4 = tf.layers.conv2d(inputs=bn3, filters=64, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=None, name='conv4H')

    # bn4 output is of size M/4 x N/4 x 64
    bn4 = tf.layers.batch_normalization(conv4, name='bn4H')

    # bn4 output is of size M/4 x N/4 x 64
    bn4 = tf.nn.relu(bn4, name='relu4H')

    # conv5 output is of size M/8 x N/8 x 128
    conv5 = tf.layers.conv2d(inputs=bn4, filters=128, kernel_size=[7, 7], strides=(2, 2), padding="same", activation=None, name='conv5H')

    # bn5 output is of size M/8 x N/8 x 128
    bn5 = tf.layers.batch_normalization(conv5, name='bn5H')

    # bn5 output is of size M/8 x N/8 x 128
    bn5 = tf.nn.relu(bn5, name='relu5H')

    # conv6 output is of size M/8 x N/8 x 128
    conv6 = tf.layers.conv2d(inputs=bn5, filters=128, kernel_size=[7, 7], strides=(1, 1), padding="same", activation=None, name='conv6H')

    # bn6 output is of size M/8 x N/8 x 128
    bn6 = tf.layers.batch_normalization(conv6, name='bn6H')

    # bn6 output is of size M/8 x N/8 x 128
    bn6 = tf.nn.relu(bn6, name='relu6H')

    # conv7 output is of size M/16 x N/16 x 128
    conv7 = tf.layers.conv2d(inputs=bn6, filters=128, kernel_size=[7, 7], strides=(2, 2), padding="same", activation=None, name='conv7H')

    # bn7 output is of size M/16 x N/16 x 128
    bn7 = tf.layers.batch_normalization(conv7, name='bn7H')

    # bn7 output is of size M/16 x N/16 x 128
    bn7 = tf.nn.relu(bn7, name='relu7H')

    # flat is of size 1 x M/16*N/16*128
    flat = tf.reshape(bn7, [-1, ImageSize[0]*ImageSize[1]*128/(16*16)], name='flatH')

    # flatdrop is a dropout layer
    flatdrop = tf.layers.dropout(flat, rate=0.75, name='flatdropH')

    # fc1 
    fc1 = tf.layers.dense(flatdrop, units=128, activation=None, name='fc1H')

    # fc1
    fc1 = tf.nn.relu(fc1, name='relu7H')

    # fc2
    fc2 = tf.layers.dense(fc1, units=8, activation=None, name='fc2H')

    return fc2

