import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def EVSegNet(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    # Img is of size MxNx3
    
    # ----------------------------------------------------
    #               ENCODER
    # ----------------------------------------------------

    # conv1 output is of size M/2 x N/2 x 8
    conv1 = tf.layers.conv2d(inputs=Img, filters=8, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv1T')

    # bn1 output is of size M/2 x N/2 x 8
    bn1 = tf.layers.batch_normalization(conv1, name='bn1T')

    # bn1 output is of size M/2 x N/2 x 8
    bn1 = tf.nn.relu(bn1, name='relu1T')

    # conv2 output is of size M/4 x N/4 x 16
    conv2 = tf.layers.conv2d(inputs=bn1, filters=16, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv2T')

    # bn2 output is of size M/4 x N/4 x 16
    bn2 = tf.layers.batch_normalization(conv2, name='bn2T')

    # bn2 output is of size M/4 x N/4 x 16
    bn2 = tf.nn.relu(bn2, name='relu2T')

    # conv3 output is of size M/8 x N/8 x 32
    conv3 = tf.layers.conv2d(inputs=bn2, filters=32, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv3T')

    # bn3 output is of size M/8 x N/8 x 32
    bn3 = tf.layers.batch_normalization(conv3, name='bn3T')

    # bn3 output is of size M/8 x N/8 x 32
    bn3 = tf.nn.relu(bn3, name='relu3T')

    
    # ----------------------------------------------------
    #               DECODER
    # ----------------------------------------------------

    # deconv1 output is of size M/4 x N/4 x 16
    deconv1 = tf.layers.conv2d_transpose(inputs=bn3, filters=16, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='deconv1T')

    # bn4 output is of size M/4 x N/4 x 16
    bn4 = tf.layers.batch_normalization(deconv1, name='bn4T')

    # bn4 output is of size M/4 x N/4 x 16
    bn4 = tf.nn.relu(bn4, name='relu4T')

    # deconv2 output is of size M/2 x N/2 x 8
    deconv2 = tf.layers.conv2d_transpose(inputs=bn4, filters=8, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='deconv2T')

    # bn5 output is of size M/2 x N/2 x 8
    bn5 = tf.layers.batch_normalization(deconv2, name='bn5T')

    # bn5 output is of size M/2 x N/2 x 8
    bn5 = tf.nn.relu(bn5, name='relu5T')

    # deconv3 output is of size M x N x 1
    deconv3 = tf.layers.conv2d_transpose(inputs=bn5, filters=2, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='deconv3T')

    # try softmax and stuff

    return deconv3

