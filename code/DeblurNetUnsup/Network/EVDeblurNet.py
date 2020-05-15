import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def EVDeblurNet(Img, PatchSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    
    # Img is of size MxNx3

    # ------------------------------------------------------------
    #                  ENCODER                                   #
    # ------------------------------------------------------------
    
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

    # ------------------------------------------------------------
    #                  DECODER                                   #
    # ------------------------------------------------------------

    # deconv1 output is of size M/8 x N/8 x 128
    deconv1 = tf.layers.conv2d_transpose(inputs=bn4, filters=128, kernel_size=[7, 7], strides=(2, 2), padding="same", activation=None, name='deconv1T')
    
    # bn5 output is of size M/8 x N/8 x 128
    bn5 = tf.layers.batch_normalization(deconv1, name='bn5T')
    
    # bn5 output is of size M/8 x N/8 x 128
    bn5 = tf.nn.relu(bn5, name='relu5T')
    
    # deconv2 output is of size M/4 x N/4 x 64
    deconv2 = tf.layers.conv2d_transpose(inputs=bn5, filters=64, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='deconv2T')
    
    # bn6 output is of size M/4 x N/4 x 64
    bn6 = tf.layers.batch_normalization(deconv2, name='bn6T')
   
    # bn6 output is of size M/4 x N/4 x 64
    bn6 = tf.nn.relu(bn6, name='relu6T')
    
    # deconv3 output is of size M/2 x N/2 x 32
    deconv3 = tf.layers.conv2d_transpose(inputs=bn6, filters=32, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='deconv3T')
    
    # bn7 output is of size M/2 x N/2 x 32
    bn7 = tf.layers.batch_normalization(deconv3, name='bn7T')
   
    # bn7 output is of size M/2 x N/2 x 32
    bn7 = tf.nn.relu(bn7, name='relu7T')
    
    # deconv4 output is of size M x N x 16
    deconv4 = tf.layers.conv2d_transpose(inputs=bn7, filters=16, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='deconv4T')
    
    # bn8 output is of size M x N x 16
    bn8 = tf.layers.batch_normalization(deconv4, name='bn8T')
    
    # bn8 output is of size M x N x 16
    bn8 = tf.nn.relu(bn8, name='relu8T')

    # deconv5 output is of size M x N x 3
    deconv5 = tf.layers.conv2d_transpose(inputs=bn8, filters=3, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None, name='deconv5T')
    
    return deconv5

