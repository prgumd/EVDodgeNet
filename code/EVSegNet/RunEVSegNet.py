#!/usr/bin/env python

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (sudo)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)

# TODO:
# Clean print statements
# Global step only loss/epoch on tensorboard
# Print Num parameters in model as a function
# Clean comments
# Check Factor from network list
# ClearLogs command line argument
# Adapt more augmentation from: https://github.com/sthalles/deeplab_v3/blob/master/preprocessing/inception_preprocessing.py
# Tensorboard logging of images

import tensorflow as tf
import cv2
import sys
import os
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.EVSegNet import EVSegNet
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(ReadPath):
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    DirNames - Full path to all image files without extension
    Train/Val/Test - Idxs of all the images to be used for training/validation (held-out testing in this case)/testing
    Ratios - Ratios is a list of fraction of data used for [Train, Val, Test]
    CheckPointPath - Path to save checkpoints/model
    OptimizerParams - List of all OptimizerParams: depends on Optimizer
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrain/Val/TestSamples - length(Train/Val/Test)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    Train/Val/TestLabels - Labels corresponding to Train/Val/Test
    """
    # Setup DirNames
    DirNamesPath = ReadPath + os.sep + 'DirNames.txt'
    TrainPath = ReadPath + os.sep + 'Train.txt'
    DirNames, TrainNames = ReadDirNames(DirNamesPath, TrainPath)
    
    # Image Input Shape
    PatchSize = np.array([260, 346, 3])
    ImageSize = np.array([260, 346, 3])
    NumTrainSamples = len(TrainNames)
    NumImgsStack = 2 # How many images to stack?
    
    return TrainNames, ImageSize, PatchSize, NumTrainSamples, NumImgsStack

def GenerateRandPatch(I, PatchSize, ImageSize=None, Vis=False):
    """
    Inputs: 
    I is the input image
    Vis when enabled, Visualizes the image and the perturbed image 
    Outputs:
    IPatch
    Points are labeled as:
    
    Top Left = p1, Top Right = p2, Bottom Right = p3, Bottom Left = p4 (Clockwise from Top Left)
    Code adapted from: https://github.com/mez/deep_homography_estimation/blob/master/Dataset_Generation_Visualization.ipynb
    """

    if(ImageSize is None):
        ImageSize = np.shape(I) 
    
    RandX = random.randint(0, ImageSize[1]-PatchSize[1])
    RandY = random.randint(0, ImageSize[0]-PatchSize[0])

    p1 = (RandX, RandY)
    p2 = (RandX, RandY + PatchSize[0])
    p3 = (RandX + PatchSize[1], RandY + PatchSize[0])
    p4 = (RandX + PatchSize[1], RandY)

    AllPts = [p1, p2, p3, p4]

    if(Vis is True):
        IDisp = I.copy()
        cv2.imshow('org', I)
        cv2.waitKey(1)

    if(Vis is True):
        IDisp = I.copy()
        cv2.polylines(IDisp, np.int32([AllPts]), 1, (0,0,0))
        cv2.imshow('a', IDisp)
        cv2.waitKey(1)

    
    Mask = np.zeros(np.shape(I))
    Mask[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :] = 1
    IPatch = I[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :]
    

    if(Vis is True):
        CroppedIDisp = np.hstack((CroppedI, CroppedWarpedI))
        print(np.shape(CroppedIDisp))
        cv2.imshow('d', CroppedIDisp)
        cv2.waitKey(0)


    # I is the Original Image I1
    # IPatch is I1 cropped to patch Size
    # AllPts is the patch corners in I1
    # Mask is the active region of I1Patch in I1
    return I, IPatch, AllPts, Mask

def PerturbImage(I1, PerturbNum):
    """
    Data Augmentation
    Inputs: 
    I1 is the input image
    PerturbNum choses type of Perturbation where it ranges from 0 to 5
    0 - No perturbation
    1 - Random gaussian Noise
    2 - Random hue shift
    3 - Random saturation shift
    4 - Random gamma shift
    Outputs:
    Perturbed Image I1
    """
    if(PerturbNum == 0):
        pass
    elif(PerturbNum == 1):
        I1 = iu.GaussianNoise(I1)
    elif(PerturbNum == 2):
        I1 = iu.ShiftHue(I1)
    elif(PerturbNum == 3):
        I1 = iu.ShiftSat(I1)
    elif(PerturbNum == 4):
        I1 = iu.Gamma(I1)
        
    return I1

def ReadDirNames(DirNamesPath, TrainPath):
    """
    Inputs: 
    Path is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read DirNames file
    DirNames = open(DirNamesPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()

    # Read TestIdxs file
    TrainIdxs = open(TrainPath, 'r')
    TrainIdxs = TrainIdxs.read()
    TrainIdxs = TrainIdxs.split()
    TrainIdxs = [int(val) for val in TrainIdxs]
    TrainNames = [DirNames[i] for i in TrainIdxs]

    return DirNames, TrainNames

def GenerateBatch(IBuffer, PatchSize):
    """
    Inputs: 
    DirNames - Full path to all image files without extension
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of I1 images after standardization and cropping/resizing to ImageSize
    HomeVecBatch - Batch of Homing Vector labels
    """
    IBatch = []

    # Generate random image
    IBuffer = np.hsplit(IBuffer, 2)
    I1 = IBuffer[0]
    I2 = IBuffer[1]
    # I = IBuffer

    # Homography and Patch generation 
    IPatch = np.dstack((I1, I2))
    # IOriginal, IPatch, AllPts, Mask = GenerateRandPatch(I, PatchSize, Vis=False)
    
    # Normalize Dataset
    # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
    IS = iu.StandardizeInputs(np.float32(IPatch))
    
    # Append All Images and Mask
    IBatch.append(IS)

    # IBatch is the Original Image I1 Batch
    return IBatch

            
def TestOperation(PatchPH, PatchSize, ModelPath, ReadPath, WritePath, TrainNames, NumTrainSamples):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    HomingVecPH is the ground truth  homing vector placeholder
    NumTrainSamples - length(Train)
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    prImg = EVDodgeNet(PatchPH, PatchSize, 1)
    prImgSoftMax = tf.nn.softmax(prImg)

    # Setup Saver
    Saver = tf.train.Saver()

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))            

        for dirs in next(os.walk(ReadPath))[1]:
            for dir in tqdm(dirs):
                CurrReadPath = ReadPath + os.sep + dir + os.sep
                NumFilesInCurrDir = len(glob.glob(CurrReadPath + 'events' + os.sep + '*.png'))
                I = cv2.imread(CurrReadPath + 'events' + os.sep + "event_%d"%1 + '.png')
                CurrWritePath = WritePath + os.sep + dir
                # Create Write Folder if doesn't exist
                if(not os.path.exists(CurrWritePath + os.sep + 'events')):
                    os.makedirs(CurrWritePath + os.sep + 'events')

                for ImgNum in tqdm(range(0, NumFilesInCurrDir)):
                    INow = cv2.imread(CurrReadPath + 'events' + os.sep + "event_%d"%(ImgNum+1) + '.png') # +1 is the offset as Image Numbers start from 1
                    IBatch = GenerateBatch(INow, PatchSize)
                    
                    FeedDict = {PatchPH: IBatch}
                    prImgVal = sess.run(prImgSoftMax, FeedDict)
                    
                    prImgVal = prImgVal[0]
                    prImgVal = prImgVal[:, :, 0:1]

                    # Remap to [0,255] range
                    prImgVal = np.uint8(remap(prImgVal, 0.0, 255.0, 0.0, 1.0)) 
                    # Crop out junk pixels as they are appended in top left corner due to padding
                    prImgVal = prImgVal[-PatchSize[0]:, -PatchSize[1]:, :]
                    
                    # Write Image to file
                    cv2.imwrite(CurrWritePath + os.sep + 'events' + os.sep +  "event_%d"%(ImgNum+1) + '.png', prImgVal)
                    
                    
def main():
    """
    Inputs: 
    None
    Outputs:
    Runs Testing code
    """
    # TODO: Make LogDir
    # TODO: Display time to end and cleanup other print statements with color
    # TODO: Make logging file a parameter

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/media/nitin/Research/EVDodge/CheckpointsEVDBHomographyDodgeNetLR5e-4Epoch200/199model.ckpt',\
                                                         help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--ReadPath', dest='ReadPath', default='/media/nitin/Research/EVDodge/DatasetChethanEvents/DeblurredHomography',\
                                                                             help='Path to load images from, Default:ReadPath')
    Parser.add_argument('--WritePath', dest='WritePath', default='/media/nitin/Research/EVDodge/DatasetChethanEvents/DeblurredHomographyDodgeNet',\
                                                                             help='Path to write images to, Default:WritePath')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    ReadPath = Args.ReadPath
    WritePath = Args.WritePath
    GPUDevice = Args.GPUDevice

    # Set GPUDevice
    tu.SetGPU(GPUDevice)

    # Setup all needed parameters including file reading
    TrainNames, ImageSize, PatchSize, NumTrainSamples, NumImgsStack = SetupAll(ReadPath)
     
    # Define PlaceHolder variables for Input and Predicted output
    PatchPH = tf.placeholder(tf.float32, shape=(1, PatchSize[0], PatchSize[1], 2*PatchSize[2]), name='Input')

    TestOperation(PatchPH, PatchSize, ModelPath, ReadPath, WritePath, TrainNames, NumTrainSamples)
     
if __name__ == '__main__':
    main()
 
