import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import tensorflow as tf
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CenterCrop(I, OutShape):
    ImageSize = np.shape(I)
    CenterX = ImageSize[0]/2
    CenterY = ImageSize[1]/2
    ICrop = I[int(np.ceil(CenterX-OutShape[0]/2)):int(np.ceil(CenterX+OutShape[0]/2)),\
              int(np.ceil(CenterY-OutShape[1]/2)):int(np.ceil(CenterY+OutShape[1]/2)), :]
    return ICrop

def CenterCropFactor(I, Factor):
    ImageSize = np.shape(I)
    CenterX = ImageSize[0]/2
    CenterY = ImageSize[1]/2
    OutShape = ImageSize - (np.mod(ImageSize,2**Factor))
    OutShape[2] = ImageSize[2]
    ICrop = I[int(np.ceil(CenterX-OutShape[0]/2)):int(np.ceil(CenterX+OutShape[0]/2)),\
              int(np.ceil(CenterY-OutShape[1]/2)):int(np.ceil(CenterY+OutShape[1]/2)), :]
    return (ICrop, OutShape)

def RandomCrop(I1, OutShape):
    ImageSize = np.shape(I1)
    RandX = random.randint(0, ImageSize[0]-OutShape[0])
    RandY = random.randint(0, ImageSize[1]-OutShape[1])
    I1Crop = I1[RandX:RandX+OutShape[0], RandY:RandY+OutShape[1], :]
    return (I1Crop)

def GaussianNoise(I1):
    IN1 = skimage.util.random_noise(I1, mode='gaussian', var=0.01)
    IN1 = np.uint8(IN1*255)
    return (IN1)

def ShiftHue(I1):
    IHSV1 = cv2.cvtColor(I1, cv2.COLOR_BGR2HSV)
    MaxShift = 30
    RandShift = random.randint(-MaxShift, MaxShift)
    IHSV1[:, :, 0] = IHSV1[:, :, 0] + RandShift
    IHSV1 = np.uint8(np.clip(IHSV1, 0, 255))
    return (cv2.cvtColor(IHSV1, cv2.COLOR_HSV2BGR))

def ShiftSat(I1):
    IHSV1 = cv2.cvtColor(I1, cv2.COLOR_BGR2HSV)
    MaxShift = 30
    RandShift = random.randint(-MaxShift, MaxShift)
    IHSV1 = np.int_(IHSV1)
    IHSV1[:, :, 1] = IHSV1[:, :, 1] + RandShift
    IHSV1 = np.uint8(np.clip(IHSV1, 0, 255))
    return (cv2.cvtColor(IHSV1, cv2.COLOR_HSV2BGR))

def Gamma(I1):
    MaxShift = 2.5
    RandShift = random.uniform(0, MaxShift)
    IG1 = skimage.exposure.adjust_gamma(I1, RandShift)
    return (IG1)

def Resize(I1, OutShape):
    ImageSize = np.shape(I1)
    I1Resize = cv2.resize(I1, (OutShape[0], OutShape[1]))
    return (I1Resize)

def StandardizeInputs(I):
    I /= 255.0
    I -= 0.5
    I *= 2.0
    return I

def StandardizeInputsTF(I):
    I = tf.math.multiply(tf.math.subtract(tf.math.divide(I, 255.0), 0.5), 2.0)
    return I
