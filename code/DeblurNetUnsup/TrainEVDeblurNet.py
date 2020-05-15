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
from Network.EVDeblurNet import EVDeblurNet
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
    CheckPointPath = '../Checkpoints/' # Path to save checkpoints
    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)

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
    PatchSize = np.array([128, 128, 3])
    ImageSize = np.array([260, 346, 3])
    NumTrainSamples = len(TrainNames)
    NumValSamples = len(ValNames)
    NumTestSamples = len(TestNames)
    
    return TrainNames, ValNames, TestNames, OptimizerParams,\
        SaveCheckPoint, ImageSize, PatchSize, NumTrainSamples, NumValSamples, NumTestSamples,\
        NumTestRunsPerEpoch


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
    
    # Read Train, Val and Test Idxs
    TrainIdxs = open(TrainPath, 'r')
    TrainIdxs = TrainIdxs.read()
    TrainIdxs = TrainIdxs.split()
    TrainIdxs = [int(val) for val in TrainIdxs]
    TrainNames = [DirNames[i] for i in TrainIdxs]

    ValIdxs = open(ValPath, 'r')
    ValIdxs = ValIdxs.read()
    ValIdxs = ValIdxs.split()
    ValIdxs = [int(val) for val in ValIdxs]
    ValNames = [DirNames[i] for i in ValIdxs]

    TestIdxs = open(TestPath, 'r')
    TestIdxs = TestIdxs.read()
    TestIdxs = TestIdxs.split()
    TestIdxs = [int(val) for val in TestIdxs]
    TestNames = [DirNames[i] for i in TestIdxs]

    return DirNames, TrainNames, ValNames, TestNames

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


def GenerateBatch(TrainNames, PatchSize, MiniBatchSize, BasePath):
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
    AllPtsBatch = []
    IPatchBatch = []
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
        IOriginal, IPatch, AllPts, Mask = GenerateRandPatch(I, PatchSize, Vis=False)

        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IS = iu.StandardizeInputs(np.float32(IPatch))

        # Append All Images and Mask
        IBatch.append(IS)
        IOrgBatch.append(I)
        AllPtsBatch.append(AllPts)
        IPatchBatch.append(IPatch)
        MaskBatch.append(Mask)

        
    # IBatch is the Original Image I1 Batch
    # IPatchBatch is I1 cropped to patch Size Batch
    # AllPtsBatch is the patch corners in I1 Batch
    # MaskBatch is the active region of I1Patch in I1 Batch
    return IBatch, IOrgBatch, AllPtsBatch, IPatchBatch, MaskBatch


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

def LossFunc(prImg, PatchPH, MiniBatchSize, LossWts, LossFuncName, SymType):
    Dx, Dy = tf.image.image_gradients(prImg)
    Meanx, Variancex = tf.nn.moments(Dx, axes=[0, 1, 2], shift=None, name='Moments', keep_dims=True)
    Meany, Variancey = tf.nn.moments(Dy, axes=[0, 1, 2], shift=None, name='Moments', keep_dims=True)

    # Photometric Loss for Identity Mapping
    if(SymType =='L1'):
        lossPhoto = tf.reduce_mean(LossWts[1]*tf.abs(prImg - PatchPH))
    elif(SymType == 'Chab'):
        epsilon = 1e-3
        alpha = 0.45
        lossPhoto = tf.reduce_mean(tf.pow(tf.square(prImg - PatchPH) + tf.square(epsilon), alpha))

    # Contrast Loss for Deblurring
    if(LossFuncName == 'V'):
        # Variance of Gradient + Photometric
        lossContrast =  - tf.reduce_mean(-LossWts[0]*(tf.abs(Variancex)+tf.abs(Variancey)))
    elif(LossFuncName == 'M'):        
        # Mean of Gradient + Photometric
        lossContrast = - tf.reduce_mean(-LossWts[0]*(tf.abs(Meanx)+tf.abs(Meany)))
        
    loss = lossPhoto + lossContrast
    return loss
    
def TrainOperation(PatchPH, IPH, MaskPH, TrainNames, TestNames, NumTrainSamples, ImageSize, PatchSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch,
                   DivTrain, LatestFile, LossFuncName, BasePath, LogsPath, SymType):
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
    prImg = EVDeblurNet(PatchPH, PatchSize, MiniBatchSize)

    with tf.name_scope('Loss'):
        LossWts = [10.0, 1.0] # First for Mean/Var, Second for Identity 2.0, 1.0
        loss = LossFunc(prImg, PatchPH, MiniBatchSize, LossWts, LossFuncName, SymType)
    
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
    tf.summary.image('PredImg', prImg[:,:,:,0:3])
    tf.summary.image('IPatch', PatchPH[:,:,:,0:3])
    tf.summary.image('I', IPH[:,:,:,0:3])
    tf.summary.image('Mask', MaskPH[:,:,:,0:3])
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

                IBatch, IOrgBatch, AllPtsBatch, IPatchBatch, MaskBatch = GenerateBatch(TrainNames, PatchSize, MiniBatchSize, BasePath)

                FeedDict = {PatchPH: IBatch, IPH: IOrgBatch, MaskPH: MaskBatch}
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
    Parser.add_argument('--BasePath', default='/media/nitin/Research/EVDodge/downfacing_processed', help='Base path of images, Default:/media/nitin/Research/EVDodge/downfacing_processed')
    Parser.add_argument('--NumEpochs', type=int, default=200, help='Number of Epochs to Train for, Default:200')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=256, help='Size of the MiniBatch to use, Default:256')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--CheckPointPath', default='../CheckpointsHomography/', help='Path to save checkpoints, Default:../CheckpointsHomography/')
    Parser.add_argument('--LogsPath', default='/media/nitin/Research/EVDodge/Logs/', help='Path to save Logs, Default:/media/nitin/Research/EVDodge/Logs/')
    Parser.add_argument('--LossFuncName', default='M', help='Choice of Loss functions, choose from M for Mean, V for Variance, Default:M')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--LR', type=float, default=1e-4, help='Learning Rate, Default: 1e-4')
    Parser.add_argument('--SymType', default='L1', help='Similarity mapping, choose from L1 and Chab, Default:L1')
    
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    LossFuncName = Args.LossFuncName
    GPUDevice = Args.GPUDevice
    LearningRate = Args.LR
    SymType = Args.SymType
    
    # Set GPUDevice
    tu.SetGPU(GPUDevice)


    # Setup all needed parameters including file reading
    TrainNames, ValNames, TestNames, OptimizerParams,\
    SaveCheckPoint, ImageSize, PatchSize, NumTrainSamples, NumValSamples, NumTestSamples,\
    NumTestRunsPerEpoch = SetupAll(BasePath, LearningRate)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, NumTestSamples, LatestFile)
        
    # Define PlaceHolder variables for Input and Predicted output
    PatchPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='Input')

    # PH for losses
    IPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]), name='IPH')
    MaskPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]), name='Mask')

    TrainOperation(PatchPH, IPH, MaskPH, TrainNames, TestNames, NumTrainSamples, ImageSize, PatchSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch,
                   DivTrain, LatestFile, LossFuncName, BasePath, LogsPath, SymType)
        
    
if __name__ == '__main__':
    main()

