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
from Network.EVHomographyNetUnsupSmall import EVHomographyNetUnsupSmall
from Network.EVHomographyNetUnsup import EVHomographyNetUnsup
from Network.EVHomographyNetUnsupSmallRobust import EVHomographyNetUnsupSmallRobust
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
import Misc.SpecUtils as su
import Misc.STNUtils as stn
import Misc.TFUtils as tu

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
    PatchSize = np.array([128, 128, 3])
    ImageSize = np.array([260, 346, 3])
    NumTrainSamples = len(TrainNames)
    
    return TrainNames, ImageSize, PatchSize, NumTrainSamples

def GenerateRandPatch(I, Rho, PatchSize, CropType, ImageSize=None, Vis=False):
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
    
    CenterX = PatchSize[1]/2
    CenterY = PatchSize[0]/2
    if(CropType == 'C'):
        RandX = int(np.floor(ImageSize[1]/2 - PatchSize[1]/2))
        RandY = int(np.floor(ImageSize[0]/2 - PatchSize[0]/2))
    elif(CropType == 'R'):
         RandX = int(random.randint(0, ImageSize[1]-PatchSize[1]))
         RandY = int(random.randint(0, ImageSize[0]-PatchSize[0]))
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
        cv2.polylines(IDisp, np.int32([AllPts]), 1, (255,255,255))
        cv2.imshow('a', IDisp)
        cv2.waitKey(1)
        
    PerturbPts = []
    for point in AllPts:
        if(len(Rho) == 1):
            # If only 1 value of Rho is given, perturb by [-Rho, Rho]
            PerturbPts.append((point[0] + random.randint(-Rho[0],Rho[0]), point[1] + random.randint(-Rho[0],Rho[0])))
        elif(len(Rho) == 2):
            if(Rho[0] != Rho[1]):
                # If bounds on Rho are given, perturb by a random value in [Rho1, Rho2] union [-Rho2, -Rho1] if Rho2 > Rho1
                PerturbPts.append((point[0] + random.choice(range(Rho[0], Rho[1]+1))*random.choice([-1, 1]),\
                                   point[1] + random.choice(range(Rho[0], Rho[1]+1))*random.choice([-1, 1])))
            else:
                # If Rho1 = Rho2 (Perturb with that amount)
                PerturbPts.append((point[0] + Rho[0], point[1] + Rho[0]))

    if(Vis is True):
        PertubImgDisp = I.copy()
        cv2.polylines(PertubImgDisp, np.int32([PerturbPts]), 1, (255,255,255))
        cv2.imshow('b', PertubImgDisp)
        cv2.waitKey(1)
        
    # Obtain Homography between the 2 images
    H = cv2.getPerspectiveTransform(np.float32(AllPts), np.float32(PerturbPts))
    # Get Inverse Homography
    HInv = np.linalg.inv(H)

    WarpedI = cv2.warpPerspective(I, HInv, (ImageSize[1],ImageSize[0]))
    if(Vis is True):
        WarpedImgDisp = WarpedI.copy()
        cv2.imshow('c', WarpedImgDisp)
        cv2.waitKey(1)

    Mask = np.zeros(np.shape(I))
    Mask[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :] = 1
    I1Patch = I[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :]
    I2Patch = WarpedI[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :]


    if(Vis is True):
        CroppedIDisp = np.hstack((I1Patch, I2Patch))
        # print(np.shape(CroppedIDisp))
        cv2.imshow('d', CroppedIDisp)
        cv2.waitKey(0)

    H4Pt = np.subtract(np.array(PerturbPts), np.array(AllPts))
    H4PtCol = np.reshape(H4Pt, (np.product(H4Pt.shape), 1))

    # I is the Original Image I1
    # IPatch is I1 cropped to patch Size
    # AllPts is the patch corners in I1
    # Mask is the active region of I1Patch in I1
    return I, I1Patch, I2Patch, AllPts, PerturbPts, H4PtCol, Mask


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

def GenerateBatch(IBuffer, Rho, PatchSize, CropType, Vis=False):
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
    I1Batch = []
    I2Batch = []
    AllPtsBatch = []
    PerturbPtsBatch = []
    H4PtColBatch = []
    MaskBatch = []

    # Generate random image
    if(np.shape(IBuffer)[1]>346):
        IBuffer = np.hsplit(IBuffer, 2)
        I1 = IBuffer[0]
    else:
        I1 = IBuffer

    # Homography and Patch generation
    IOriginal, I1Patch, I2Patch, AllPts, PerturbPts,\
    H4PtCol, Mask = GenerateRandPatch(I1, Rho, PatchSize, CropType, Vis=Vis) # Rand Patch will take the whole image as it doesn't have a choice
    ICombo = np.dstack((I1Patch, I2Patch))

    
    # Normalize Dataset
    # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
    IS = iu.StandardizeInputs(np.float32(ICombo))
    
    # Append All Images and Mask
    IBatch.append(IS)
    I1Batch.append(I1Patch)
    I2Batch.append(I2Patch)
    AllPtsBatch.append(AllPts)
    PerturbPtsBatch.append(PerturbPts)
    H4PtColBatch.append(H4PtCol)
    MaskBatch.append(MaskBatch)

    # IBatch is the Original Image I1 Batch
    return IBatch, I1Batch, I2Batch, AllPtsBatch, PerturbPtsBatch, H4PtColBatch, MaskBatch

            
def TestOperation(PatchPH, I1PH, I2PH, PatchSize, ModelPath, ReadPath, WritePath, TrainNames, NumTrainSamples, CropType):
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
    # Generate indexes for center crop of train size
    CenterX = PatchSize[1]/2
    CenterY = PatchSize[0]/2
    RandX = np.ceil(CenterX - PatchSize[1]/2)
    RandY = np.ceil(CenterY - PatchSize[0]/2)
    p1 = (RandX, RandY)
    p2 = (RandX, RandY + PatchSize[0])
    p3 = (RandX + PatchSize[1], RandY + PatchSize[0])
    p4 = (RandX + PatchSize[1], RandY)
    
    AllPts = [p1, p2, p3, p4]
    
    # Predict output with forward pass, MiniBatchSize for Test is 1
    prHVal = EVHomographyNetUnsupSmall(PatchPH, PatchSize, 1)
    prHVal = tf.reshape(prHVal, (-1, 8, 1))
    
    HMat = stn.solve_DLT(1, AllPts, prHVal)
   
    # Warp I1 to I2
    out_size = [128, 128]
    WarpI1 = stn.transform(out_size, HMat, 1, I1PH)
    
    # Setup Saver
    Saver = tf.train.Saver()
    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        tu.FindNumParams(1)
        PredOuts = open(WritePath + os.sep + 'PredOuts.txt', 'w')
        for dirs in tqdm(next(os.walk(ReadPath))[1]):
            CurrReadPath = ReadPath + os.sep + dirs
            NumFilesInCurrDir = len(glob.glob(CurrReadPath + os.sep + '*.png'))
            I = cv2.imread(CurrReadPath + os.sep + "event_%d"%1 + '.png')
            CurrWritePath = WritePath + os.sep + dirs
            # Create Write Folder if doesn't exist
            if(not os.path.exists(CurrWritePath)):
                os.makedirs(CurrWritePath)
                
            for ImgNum in tqdm(range(0, NumFilesInCurrDir)):
                # for StackNum in range(0, NumImgsStack):
                INow = cv2.imread(CurrReadPath + os.sep + "event_%d"%(ImgNum+1) + '.png')
                Rho = [25]# [10, 10]
                IBatch, I1Batch, I2Batch, AllPtsBatch, PerturbPtsBatch, H4PtColBatch, MaskBatch = GenerateBatch(INow, Rho, PatchSize, CropType, Vis=False)
                # TODO: Better way is to feed data into a MiniBatch and Extract it again
                # INow = np.hsplit(INow, 2)[0] # Imgs have a stack of 2 in this case, hence extract one
                
                FeedDict = {PatchPH: IBatch, I1PH: I1Batch, I2PH: I2Batch}
                prHPredVal = sess.run(prHVal, FeedDict)
                prHTrue = np.float32(np.reshape(H4PtColBatch[0], (-1, 4, 2)))[0]
                ErrorNow = np.sum(np.sqrt((prHPredVal[:, 0] - prHTrue[:, 0])**2 + (prHPredVal[:, 1] - prHTrue[:, 1])**2))/4
                # print(ErrorNow)
                # print(prHPredVal)
                # print(prHTrue)
                # a = input('a')
                # Timer1 = tic()
                # WarpI1Ret = sess.run(WarpI1, FeedDict)
                # cv2.imshow('a', WarpI1Ret[0])
                # cv2.imshow('b', I1Batch[0])
                # cv2.imshow('c', I2Batch[0])
                # cv2.imshow('d', np.abs(WarpI1Ret[0]- I2Batch[0]))
                # cv2.waitKey(0)
                
                # print(toc(Timer1))
                
                # WarpI1Ret = WarpI1Ret[0]
                # Remap to [0,255] range
                # WarpI1Ret = np.uint8(remap(WarpI1Ret, 0.0, 255.0, np.amin(WarpI1Ret), np.amax(WarpI1Ret)))
                # Crop out junk pixels as they are appended in top left corner due to padding
                # WarpI1Ret = WarpI1Ret[-PatchSize[0]:, -PatchSize[1]:, :]
                
                # IStacked = np.hstack((WarpI1Ret, I2Batch[0]))
                # Write Image to file
                # cv2.imwrite(CurrWritePath + os.sep + 'events' + os.sep +  "event_%d"%(ImgNum+1) + '.png', IStacked)
                PredOuts.write(dirs + os.sep + "event_%d"%ImgNum + '.png' + '\t' + str(ErrorNow) + '\n')
        PredOuts.close()
                    
                    
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
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/media/nitin/Research/EVDodge/CheckpointsDeblurHomographyLR1e-4Epochs400/399model.ckpt',\
                                                         help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--ReadPath', dest='ReadPath', default='/media/nitin/Research/EVDodge/DatasetChethanEvents/Deblurred',\
                                                                             help='Path to load images from, Default:ReadPath')
    Parser.add_argument('--WritePath', dest='WritePath', default='/media/nitin/Research/EVDodge/DatasetChethanEvents/DeblurredHomography',\
                                                                             help='Path to load images from, Default:WritePath')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--CropType', dest='CropType', default='C', help='What kind of crop do you want to perform? R: Random, C: Center, Default: C')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    ReadPath = Args.ReadPath
    WritePath = Args.WritePath
    GPUDevice = Args.GPUDevice
    CropType = Args.CropType
    
    # Set GPUNum
    tu.SetGPU(GPUDevice)

    # Setup all needed parameters including file reading
    TrainNames, ImageSize, PatchSize, NumTrainSamples = SetupAll(ReadPath)
     
    # Define PlaceHolder variables for Input and Predicted output
    PatchPH = tf.placeholder(tf.float32, shape=(1, PatchSize[0], PatchSize[1], PatchSize[2]*2), name='Input')
    I1PH = tf.placeholder(tf.float32, shape=(1, PatchSize[0], PatchSize[1], PatchSize[2]), name='I1')
    I2PH = tf.placeholder(tf.float32, shape=(1, PatchSize[0], PatchSize[1], PatchSize[2]), name='I2')

    if(not os.path.exists(WritePath)):
        cprint("WARNING: %s doesnt exist, Creating it."%WritePath, 'yellow')
        os.mkdir(WritePath)

    TestOperation(PatchPH, I1PH, I2PH, PatchSize, ModelPath, ReadPath, WritePath, TrainNames, NumTrainSamples, CropType)
     
if __name__ == '__main__':
    main()
 
