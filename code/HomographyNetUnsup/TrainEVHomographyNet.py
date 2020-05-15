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

def SetupAll(BasePath, LearningRate):
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
    DirNamesPath = BasePath + os.sep + 'DirNames.txt'
    # LabelNamesPath = BasePath + os.sep + 'Labels.txt'
    TrainPath = BasePath + os.sep + 'Train.txt'
    ValPath = BasePath + os.sep + 'Val.txt'
    TestPath = BasePath + os.sep + 'Test.txt'
    DirNames, TrainNames, ValNames, TestNames=\
              ReadDirNames(DirNamesPath, TrainPath, ValPath, TestPath)


    # Setup Neural Net Params
    # List of all OptimizerParams: depends on Optimizer
    # For ADAM Optimizer: [LearningRate, Beta1, Beta2, Epsilion]
    UseDefaultFlag = 0 # Set to 0 to use your own params, do not change default parameters
    if UseDefaultFlag:
        # Default Parameters
        OptimizerParams = [1e-3, 0.9, 0.999, 1e-8]
    else:
        # Custom Parameters
        OptimizerParams = [LearningRate, 0.9, 0.999, 1e-8]   
        
    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100 
    # Number of passes of Val data with MiniBatchSize 
    NumTestRunsPerEpoch = 5
    
    # Image Input Shape
    ImageSize = np.array([128, 128, 3])
    Rho = 25
    NumTrainSamples = len(TrainNames)
    NumValSamples = len(ValNames)
    NumTestSamples = len(TestNames)
    
    return TrainNames, ValNames, TestNames, OptimizerParams,\
        SaveCheckPoint, ImageSize, Rho, NumTrainSamples, NumValSamples, NumTestSamples,\
        NumTestRunsPerEpoch

def RandHomographyPerturbation(I, Rho, PatchSize, ImageSize=None, Vis=False):
    """
    Inputs: 
    I is the input image
    Rho is the maximum perturbation in either direction on each corner, i.e., perturbation for each corner lies in [-Rho, Rho]
    Vis when enabled, Visualizes the image and the perturbed image 
    Outputs:
    H is the random homography
    Points are labeled as:
    
    Top Left = p1, Top Right = p2, Bottom Right = p3, Bottom Left = p4 (Clockwise from Top Left)
    Code adapted from: https://github.com/mez/deep_homography_estimation/blob/master/Dataset_Generation_Visualization.ipynb
    """

    if(ImageSize is None):
        ImageSize = np.shape(I) 
    
    RandX = random.randint(Rho, ImageSize[1]-Rho-PatchSize[1])
    RandY = random.randint(Rho, ImageSize[0]-Rho-PatchSize[0])

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

    PerturbPts = []
    for point in AllPts:
        PerturbPts.append((point[0] + random.randint(-Rho,Rho), point[1] + random.randint(-Rho,Rho)))

    if(Vis is True):
        PertubImgDisp = I.copy()
        cv2.polylines(PertubImgDisp, np.int32([PerturbPts]), 1, (0,0,0))
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
    CroppedI = I[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :]
    CroppedWarpedI = WarpedI[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :]
    

    if(Vis is True):
        CroppedIDisp = np.hstack((CroppedI, CroppedWarpedI))
        print(np.shape(CroppedIDisp))
        cv2.imshow('d', CroppedIDisp)
        cv2.waitKey(0)

    H4Pt = np.subtract(np.array(PerturbPts), np.array(AllPts))
    H4PtCol = np.reshape(H4Pt, (np.product(H4Pt.shape), 1))

    # I is the Original Image I1
    # WarpedI is I2
    # CroppedI is I1 cropped to patch Size
    # CroppedWarpeI is I1 Crop Warped (I2 Crop)
    # AllPts is the patch corners in I1
    # PerturbPts is the patch corners of I2 in I1
    # H4PtCol is the delta movement of corners between AllPts and PerturbPts
    # Mask is the active region of I1Patch in I1
    return I, WarpedI, CroppedI, CroppedWarpedI, AllPts, PerturbPts, H4PtCol, Mask
        

def ReadDirNames(DirNamesPath, TrainPath, ValPath, TestPath):
    """
    Inputs: 
    Path is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read DirNames and LabelNames files
    DirNames = open(DirNamesPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()

    # LabelNames = open(LabelNamesPath, 'r')
    # LabelNames = LabelNames.read()
    # LabelNames = LabelNames.split()
    
    # Read Train, Val and Test Idxs
    TrainIdxs = open(TrainPath, 'r')
    TrainIdxs = TrainIdxs.read()
    TrainIdxs = TrainIdxs.split()
    TrainIdxs = [int(val) for val in TrainIdxs]
    TrainNames = [DirNames[i] for i in TrainIdxs]
    # TrainLabels = [LabelNames[i] for i in TrainIdxs]

    ValIdxs = open(ValPath, 'r')
    ValIdxs = ValIdxs.read()
    ValIdxs = ValIdxs.split()
    ValIdxs = [int(val) for val in ValIdxs]
    ValNames = [DirNames[i] for i in ValIdxs]
    # ValLabels = [LabelNames[i] for i in ValIdxs]

    TestIdxs = open(TestPath, 'r')
    TestIdxs = TestIdxs.read()
    TestIdxs = TestIdxs.split()
    TestIdxs = [int(val) for val in TestIdxs]
    TestNames = [DirNames[i] for i in TestIdxs]
    # TestLabels = [LabelNames[i] for i in TestIdxs]

    return DirNames, TrainNames, ValNames, TestNames
    
def GenerateBatch(TrainNames, ImageSize, MiniBatchSize, Rho, BasePath):
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
    IOrgBatch = []
    LabelBatch = []
    AllPtsBatch = []
    PerturbPtsBatch = []
    WarpedIBatch = []
    CroppedIBatch = []
    CroppedWarpedIBatch = []
    MaskBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainNames)-1)        
        RandImageName = BasePath + os.sep + TrainNames[RandIdx] 
        ImageNum += 1
        IBuffer = cv2.imread(RandImageName)
        if(np.shape(IBuffer)[1]>346):
            IBuffer = np.hsplit(IBuffer, 2)
            I = IBuffer[0]
        else:
            I = IBuffer

        # Homography and Patch generation 
        IOriginal, WarpedI, CroppedI, CroppedWarpedI, AllPts, PerturbPts, H4PtCol, Mask = RandHomographyPerturbation(I, Rho, ImageSize, Vis=False)

        ICombined = np.dstack((CroppedI[:,:,0:3], CroppedWarpedI[:,:,0:3]))
        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IS = iu.StandardizeInputs(np.float32(ICombined))

        # Append All Images and Mask
        IBatch.append(IS)
        LabelBatch.append(H4PtCol)
        IOrgBatch.append(I)
        AllPtsBatch.append(AllPts)
        PerturbPtsBatch.append(PerturbPts)
        WarpedIBatch.append(WarpedI)
        CroppedIBatch.append(CroppedI)
        CroppedWarpedIBatch.append(CroppedWarpedI)
        MaskBatch.append(Mask)

        
    # IBatch is the Cropped stack of I1 and I2 Batch
    # LabelBatch is the Homography Ideal Batch
    # IOrgBatch is I1 Batch
    # WarpedIBatch is I2 Batch
    # CroppedIBatch is I1Patch Batch
    # CroppedWarpedIBatch is I2Patch Batch
    # AllPtsBatch is the patch corners in I1 Batch
    # PerturbPtsBatch is the patch corners of I2 in I1 Batch
    # MaskBatch is the active region of I1Patch Batch in I1 Batch
    return IBatch, LabelBatch, IOrgBatch, WarpedIBatch, CroppedIBatch, CroppedWarpedIBatch, AllPtsBatch, PerturbPtsBatch, MaskBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, NumTestSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    print('Number of Testing Images ' + str(NumTestSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              


def RobustLoss(x, a, c, e=1e-2):
	b = tf.abs(2.-a) + e
	d = tf.where(tf.greater_equal(a, 0.), a+e, a-e)
	return b/d*(tf.pow(tf.square(x/c)/b+1., 0.5*d)-1.)

def logZ1(a):
	ps = [1.49130350, 1.38998350, 1.32393250,
	1.26937670, 1.21922380, 1.16928990,
	1.11524570, 1.04887590, 0.91893853]

	ms = [-1.4264522e-01, -7.7125795e-02, -5.9283373e-02,
	-5.2147767e-02, -5.0225594e-02, -5.2624177e-02,
	-6.1122095e-02, -8.4174540e-02, -2.5111488e-01]

	x = 8. * (tf.log(a + 2.) / tf.log(2.) - 1.)
	i0 = tf.cast(tf.clip_by_value(x, 0., tf.cast(len(ps) - 2, tf.float32)), tf.int32)
	p0 = tf.gather(ps, i0)
	p1 = tf.gather(ps, i0 + 1)
	m0 = tf.gather(ms, i0)
	m1 = tf.gather(ms, i0 + 1)
	t = x - tf.cast(i0, tf.float32)
	h01 = t * t * (-2. * t + 3.)
	h00 = 1. - h01
	h11 = t * t * (t - 1.)
	h10 = h11 + t * (1. - t)
	return tf.where(t < 0., ms[0] * t + ps[0],
	tf.where(t > 1., ms[-1] * (t - 1.) + ps[-1],
	p0 * h00 + p1 * h01 + m0 * h10 + m1 * h11))

def nll(x, a, c, e=1e-2):
	return RobustLoss(x, a, c, e) + logZ1(a) # + tf.log(c)

def LossFunc(I1PH, I2PH, MaskPH, AllPtsPH, LabelPH, prHVal, MiniBatchSize, LossFuncName, TrainingType, a=None, c=1e-1, ImageSize=None):
    prHVal = tf.reshape(prHVal, (-1, 8, 1))
    # Warp Image using predicted Homography values
    # Obtain 3x3 H matrix from 8x1 values
    HMat = stn.solve_DLT(MiniBatchSize, AllPtsPH, prHVal)
    
    # Warp I1 to align with I2
    out_size = [260, 346]
    WarpI1 = stn.transform(out_size, HMat, MiniBatchSize, I1PH)
    WarpMask = stn.transform(out_size, HMat, MiniBatchSize, MaskPH)
    
    if(TrainingType == 'US'):        
        if(LossFuncName == 'PhotoL1'):
            WarpI1Patch = tf.boolean_mask(WarpI1, MaskPH)
            I2Patch = tf.boolean_mask(I2PH, MaskPH)
            
            DiffImg = WarpI1Patch - I2Patch
            lossPhoto = tf.reduce_mean(tf.abs(DiffImg))
        elif(LossFuncName == 'PhotoChab'):
            WarpI1Patch = tf.boolean_mask(WarpI1, MaskPH)
            I2Patch = tf.boolean_mask(I2PH, MaskPH)

            DiffImg = WarpI1Patch - I2Patch
            epsilon = 1e-3
            alpha = 0.45
            lossPhoto = tf.reduce_mean(tf.pow(tf.square(DiffImg) + tf.square(epsilon), alpha))
        elif(LossFuncName == 'PhotoRobust'):
            WarpI1Patch = tf.reshape(tf.boolean_mask(WarpI1, MaskPH), [MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]])
            I2Patch = tf.reshape(tf.boolean_mask(I2PH, MaskPH), [MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]])
            
            DiffImg = WarpI1Patch - I2Patch
    	    Epsa = 1e-3
    	    a = tf.multiply((2.0 - 2.0*Epsa), tf.math.sigmoid(a)) + Epsa
            # a = tf.image.resize_images(a, out_size)
    	    lossPhoto = tf.reduce_mean(nll(DiffImg, a, c))

    elif(TrainingType == 'S'):
        WarpI1Patch = tf.boolean_mask(WarpI1, MaskPH)
        I2Patch = tf.boolean_mask(I2PH, MaskPH)
        # L2 loss between predicted and ground truth H4Pt (4 point homography)
        lossPhoto = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.squeeze(prHVal) - tf.squeeze(LabelPH)), axis=1)))
        
    
    return lossPhoto, WarpI1, WarpMask, WarpI1Patch, I2Patch, a
    
def TrainOperation(ImgPH, I1PH, I2PH, MaskPH, AllPtsPH, LabelPH, TrainNames, TestNames, NumTrainSamples, ImageSize, Rho,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch,
                   DivTrain, LatestFile, LossFuncName, NetworkType, BasePath, LogsPath, TrainingType):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    HomingVecPH is the ground truth  homing vector placeholder
    DirNames - Full path to all image files without extension
    Train/Val - Idxs of all the images to be used for training/validation (held-out testing in this case)
    Train/ValLabels - Labels corresponding to Train/Val
    NumTrain/ValSamples - length(Train/Val)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    OptimizerParams - List of all OptimizerParams: depends on Optimizer
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of data
    LatestFile - Latest checkpointfile to continue training
    Outputs:
    Saves Trained network in CheckPointPath
    """
    
    # Predict output with forward pass
    if(NetworkType == 'Small'):
        prHVal = EVHomographyNetUnsupSmall(ImgPH, ImageSize, MiniBatchSize)
        with tf.name_scope('Loss'):
    	    loss, WarpI1, WarpMask, WarpI1Patch, I2Patch, _ = LossFunc(I1PH, I2PH, MaskPH, AllPtsPH, LabelPH, prHVal, MiniBatchSize, LossFuncName, TrainingType)
    elif(NetworkType == 'Large'):
        prHVal = EVHomographyNetUnsup(ImgPH, ImageSize, MiniBatchSize)
        with tf.name_scope('Loss'):
    	    loss, WarpI1, WarpMask, WarpI1Patch, I2Patch, _ = LossFunc(I1PH, I2PH, MaskPH, AllPtsPH, LabelPH, prHVal, MiniBatchSize, LossFuncName, TrainingType)
    elif(NetworkType == 'SmallRobust'):
        prHVal, prAlpha = EVHomographyNetUnsupSmallRobust(ImgPH, ImageSize, MiniBatchSize)
        with tf.name_scope('Loss'):
    	    loss, WarpI1, WarpMask, WarpI1Patch, I2Patch, Alpha = LossFunc(I1PH, I2PH, MaskPH, AllPtsPH, LabelPH, prHVal, MiniBatchSize, LossFuncName, TrainingType, a=prAlpha, ImageSize=ImageSize)
            
    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=OptimizerParams[0], beta1=OptimizerParams[1],
                                           beta2=OptimizerParams[2], epsilon=OptimizerParams[3])
        Gradients = Optimizer.compute_gradients(loss)
        OptimizerUpdate = Optimizer.apply_gradients(Gradients)
        #Optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8).minimize(loss)
        #Optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.image('WarpedImg', WarpI1[:,:,:,0:3])
    tf.summary.image('I1', I1PH[:,:,:,0:3])
    tf.summary.image('I2', I2PH[:,:,:,0:3])
    tf.summary.image('Mask', MaskPH[:,:,:,0:3])
    tf.summary.image('WarpMask', WarpMask[:,:,:,0:3])
    # tf.summary.image('WarpI1Patch', WarpI1Patch[:,:,:,0:3])
    # tf.summary.image('I2Patch', I2Patch[:,:,:,0:3])
    # tf.summary.image('I2Patch - WarpI1Patch', tf.abs(I2Patch[:,:,:,0:3] - WarpI1Patch[:,:,:,0:3]))
    tf.summary.histogram('prHVal', prHVal, collections=None, family=None)
    if(NetworkType == 'SmallRobust'):
        tf.summary.image('Alpha1', Alpha[:,:,:,0:1])
        tf.summary.image('Alpha2', Alpha[:,:,:,1:2])
        tf.summary.image('Alpha3', Alpha[:,:,:,2:3])
        tf.summary.histogram('Alpha1', Alpha[:,:,:,0:1], collections=None, family=None)
        tf.summary.histogram('Alpha2', Alpha[:,:,:,1:2], collections=None, family=None)
        tf.summary.histogram('Alpha3', Alpha[:,:,:,2:3], collections=None, family=None)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()
    
    AllEpochLoss = [0.0]
    EachIterLoss = [0.0]
    # Setup Saver
    Saver = tf.train.Saver()
    
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Print Number of parameters in the network    
        tu.FindNumParams(1)
        
        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
            
        TotalTimeElapsed = 0.0
        TimerOverall = tic()
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            EpochLoss = 0.0
            Timer1 = tic()
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                print('Epoch ' + str(Epochs) + ' PerEpochCounter ' + str(PerEpochCounter))
                Timer2 = tic()

                IBatch, LabelBatch, I1Batch, I2Batch, I1PatchBatch, I2PatchBatch, \
                AllPtsBatch, PerturbPtsBatch, MaskBatch = GenerateBatch(TrainNames, ImageSize, MiniBatchSize, Rho, BasePath)

                FeedDict = {ImgPH: IBatch, I1PH: I1Batch, AllPtsPH: AllPtsBatch, I2PH: I2Batch, MaskPH: MaskBatch, LabelPH: LabelBatch}
                _, LossThisBatch, Summary = sess.run([OptimizerUpdate, loss, MergedSummaryOP], feed_dict=FeedDict)
                
                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()

                # Calculate and print Train accuracy (also called EpochLoss) every epoch
                EpochLoss += LossThisBatch

                # Save All losses
                EachIterLoss.append(LossThisBatch)

                TimeLastMiniBatch = toc(Timer2)

                # Print LossThisBatch
                print('LossThisBatch is  '+ str(LossThisBatch))
                
                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print(SaveName + ' Model Saved...')

                # Print timing information
                EstimatedTimeToCompletionThisEpoch = float(TimeLastMiniBatch)*float(NumIterationsPerEpoch-PerEpochCounter-1.0)
                EstimatedTimeToCompletionTotal = float(TimeLastMiniBatch)*float(NumIterationsPerEpoch-PerEpochCounter-1.0) +\
                                                 float(TimeLastMiniBatch)*float(NumIterationsPerEpoch-1.0)*float(NumEpochs-Epochs)
                TotalTimeElapsed = toc(TimerOverall)
                print('Percentage complete in total epochs ' + str(float(Epochs+1)/float(NumEpochs-StartEpoch+1)*100.0))
                print('Percentage complete in this Train epoch ' + str(float(PerEpochCounter)/float(NumIterationsPerEpoch)*100.0))
                print('Last MiniBatch took '+ str(TimeLastMiniBatch) + ' secs, time taken till now ' + str(TotalTimeElapsed) + \
                      ' estimated time to completion of this epoch is ' + str(EstimatedTimeToCompletionThisEpoch))
                print('Estimated Total time remaining is ' + str(EstimatedTimeToCompletionTotal))
                
            TimeLastEpoch = toc(Timer1)
            EstimatedTimeToCompletion = float(TotalTimeElapsed)/float(Epochs+1.0)*float(NumEpochs-Epochs-1.0)
                
            # Save Each Epoch loss
            AllEpochLoss.append(EpochLoss)
            
            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print(SaveName + ' Model Saved...')


def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # TODO: Make LogDir
    # TODO: Make logging file a parameter
    # TODO: Time to complete print

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/nitin/EVDodge/DatasetsForPaper/DownfacingEventsProcessed', help='Base path of images, Default:/home/nitin/EVDodge/DatasetsForPaper/DownfacingEventsProcessed')
    Parser.add_argument('--NumEpochs', type=int, default=200, help='Number of Epochs to Train for, Default:200')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=256, help='Size of the MiniBatch to use, Default:256')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LossFuncName', default='PhotoL1', help='Choice of Loss functions, choose from PhotoL1, PhotoChab, PhotoRobust when using TrainingType as US. Default:PhotoL1')
    Parser.add_argument('--NetworkType', default='Small', help='Choice of Network type, choose from Small, Large, Default:Small')
    Parser.add_argument('--CheckPointPath', default='../CheckpointsHomography/', help='Path to save checkpoints, Default:../CheckpointsHomography/')
    Parser.add_argument('--LogsPath', default='/media/nitin/Research/EVDodge/Logs/', help='Path to save Logs, Default:/media/nitin/Research/EVDodge/Logs/')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--LR', type=float, default=1e-4, help='Learning Rate, Default: 1e-4')
    Parser.add_argument('--TrainingType', default='US', help='Training Type, S: Supervised, US: Unsupervised, Default: US')
    
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    LossFuncName = Args.LossFuncName
    NetworkType = Args.NetworkType
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    GPUDevice = Args.GPUDevice
    LearningRate = Args.LR
    TrainingType = Args.TrainingType
    
    # Set GPUDevice
    tu.SetGPU(GPUDevice)

    # Setup all needed parameters including file reading
    TrainNames, ValNames, TestNames, OptimizerParams,\
    SaveCheckPoint, ImageSize, Rho, NumTrainSamples, NumValSamples, NumTestSamples,\
    NumTestRunsPerEpoch = SetupAll(BasePath, LearningRate)

    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, NumTestSamples, LatestFile)
        
    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 6), name='Input')

    # PH for losses
    I1PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 260, 346, 3), name='I1')
    I2PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 260, 346, 3), name='I2')
    MaskPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 260, 346, 3), name='Mask')
    AllPtsPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 4, 2), name='AllPts')
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8, 1), name='Label')  

    TrainOperation(ImgPH, I1PH, I2PH, MaskPH, AllPtsPH, LabelPH, TrainNames, TestNames, NumTrainSamples, ImageSize, Rho,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch,
                   DivTrain, LatestFile, LossFuncName, NetworkType, BasePath, LogsPath, TrainingType)
        
    
if __name__ == '__main__':
    main()

