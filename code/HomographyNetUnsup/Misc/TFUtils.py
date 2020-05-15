import os, pdb
import sys
import time
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# Don't generate pyc codes
sys.dont_write_bytecode = True

def FindNumParams(PrintFlag=None):
    if(PrintFlag is not None):
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

def SetGPU(GPUNum=-1):
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPUNum)

def Rename(CheckPointPath, ReplaceSource=None, ReplaceDestination=None, AddPrefix=None):
    # Help!
    # https://github.com/tensorflow/models/issues/974
    # https://stackoverflow.com/questions/37086268/rename-variable-scope-of-saved-model-in-tensorflow
    # https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
    # Rename to correct paths in checkpoint file
    # CheckPointPath points to folder with all the model files and checkpoint file
    if(not os.path.isdir(CheckPointPath)):
        print('CheckPointsPath should be a directory!')
        os._exit(0)
        
    CheckPoint = tf.train.get_checkpoint_state(CheckPointPath)
    with tf.Session() as sess:
        for VarName, _ in tf.contrib.framework.list_variables(CheckPointPath):
            # Load the variable
            Var = tf.contrib.framework.load_variable(CheckPointPath, VarName)

            NewName = VarName
            if(ReplaceSource is not None):
                NewName = NewName.replace(ReplaceSource, ReplaceDestination)
            if(AddPrefix is not None):
                NewName = AddPrefix + NewName

            print('Renaming %s to %s.' % (VarName, NewName))
            # Rename the variable
            Var = tf.Variable(Var, name=NewName)
            
        # Save the variables
        Saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        Saver.save(sess, CheckPoint.model_checkpoint_path)

def PrintVars(CheckPointPath):
    # Help!
    # https://github.com/tensorflow/models/issues/974
    # https://stackoverflow.com/questions/37086268/rename-variable-scope-of-saved-model-in-tensorflow
    # https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
    # Rename to correct paths in checkpoint file
    # CheckPointPath points to folder with all the model files and checkpoint file
    if(not os.path.isdir(CheckPointPath)):
        print('CheckPointsPath should be a directory!')
        os._exit(0)
        
    CheckPoint = tf.train.get_checkpoint_state(CheckPointPath)
    with tf.Session() as sess:
        for VarName, _ in tf.contrib.framework.list_variables(CheckPointPath):
            print('%s' % VarName)
