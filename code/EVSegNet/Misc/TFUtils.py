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

    
