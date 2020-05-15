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
import Misc.TFUtils as tu
import re

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
    LabelNamesPath = BasePath + os.sep + 'Labels.txt'
    TrainPath = BasePath + os.sep + 'Train.txt'
    DirNames, TrainNames, TrainLabels = ReadDirNames(DirNamesPath, LabelNamesPath, TrainPath)


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
    
    # Image Input Shape
    Factor = 3
    I = cv2.imread(BasePath + os.sep + TrainNames[0])
    _, ImageSize = iu.CenterCropFactor(I, Factor)
    NumTrainSamples = len(TrainNames)
    
    return TrainNames, TrainLabels, OptimizerParams, SaveCheckPoint, Factor, ImageSize, NumTrainSamples

def ReadDirNames(DirNamesPath, LabelNamesPath, TrainPath):
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

    LabelNames = open(LabelNamesPath, 'r')
    LabelNames = LabelNames.read()
    LabelNames = LabelNames.split()
    
    # Read Train, Val and Test Idxs
    TrainIdxs = open(TrainPath, 'r')
    TrainIdxs = TrainIdxs.read()
    TrainIdxs = TrainIdxs.split()
    TrainIdxs = [int(val) for val in TrainIdxs]
    TrainNames = [DirNames[i] for i in TrainIdxs]
    TrainLabels = [LabelNames[i] for i in TrainIdxs]

    return DirNames, TrainNames, TrainLabels
    
def GenerateBatch(TrainNames, TrainLabels, Factor, ImageSize, MiniBatchSize, BasePath, MaxFrameDiff):
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
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainNames)-MaxFrameDiff)
        RandFrameDiff = random.randint(1, MaxFrameDiff)
        
        RandImageName = TrainNames[RandIdx]
        # Create File Number in same folder with RandFrameDiff
        RandImagePairName = RandImageName.split(os.sep)[0] + os.sep + 'events' + os.sep + 'event_' + str(int(re.split('_|.png', RandImageName)[-2]) + RandFrameDiff)
        I2 = cv2.imread(BasePath + os.sep + RandImagePairName + '.png')
        if(not np.shape(I2)): # OpenCV returns empty matrix if no image is found!
            continue # Retry if RandImagePair is not valid!

        I1 = cv2.imread(BasePath + os.sep + RandImageName)

        ImageNum += 1
    
        I1, _ = iu.CenterCropFactor(I1, Factor)
        I2, _ = iu.CenterCropFactor(I2, Factor)
        
        ICombined = np.dstack((I1, I2))     

        # Standardize Inputs as given by Inception v3 paper
        # MAYBE: Find Mean of Dataset or use from ImageNet
        # MAYBE: Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IS = iu.StandardizeInputs(np.float32(ICombined))
        Label1 = cv2.imread(BasePath + os.sep + TrainLabels[RandIdx])
        Label1Name =  TrainLabels[RandIdx]
        Label2Name = Label1Name.split(os.sep)[0] + os.sep + 'masks' + os.sep + 'mask_' + '%08d.png' % (int(re.split('_|.png', RandImageName)[-2]) + RandFrameDiff) # 08
        Label2 = cv2.imread(BasePath + os.sep + Label2Name)
        LabelCropped, _ = iu.CenterCropFactor(Label1 | Label2, Factor) # Label Mask is the logical OR of both Masks
        LabelCropped = np.float32(LabelCropped[:, :, 0])/255.0
        LabelCropped = np.expand_dims(LabelCropped, axis=3)
        LabelCropped = np.dstack((LabelCropped, 1.0-LabelCropped))

        # Append All Images and Mask
        IBatch.append(IS)
        I1Batch.append(I1)
        I2Batch.append(I2)
        LabelBatch.append(LabelCropped)
        
    return IBatch, I1Batch, I2Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

def Accuracy(Pred, GT):
    """
    Inputs: 
    HomingVecPred is the output of the neural network
    HomingVecReal is the ground truth homing vector
    Outputs:
    NOT IMPLEMENTED YET! 
    """
    return (np.sum(Pred==GT)*100.0)
    
def TrainOperation(ImgPH, I1PH, I2PH, LabelPH, TrainNames, TrainLabels, NumTrainSamples, Factor, ImageSize, NumEpochs,
                   MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, DivTrain, LatestFile, MaxFrameDiff, LogsPath, BasePath):
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
    prImg = EVDodgeNet(ImgPH, ImageSize, MiniBatchSize)
    prImgSoftMax = tf.nn.softmax(prImg)

    with tf.name_scope('Loss'):
        # Cross-Entropy Loss
        # We want to evaluate Cross entropy loss which expects size of BatchSize x NumClasses
        # Hence We will reshape output to 1D
        prImgFlat = tf.reshape(prImg, (-1, 2))
        LabelPHFlat = tf.reshape(LabelPH, (-1, 2))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prImgFlat, labels=LabelPHFlat))
        
    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=OptimizerParams[0], beta1=OptimizerParams[1],
                                           beta2=OptimizerParams[2], epsilon=OptimizerParams[3]).minimize(loss)
        #Optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8).minimize(loss)
        #Optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.image('prImg0', prImgSoftMax[:,:,:,0:1])
    tf.summary.image('prImg1', prImgSoftMax[:,:,:,1:])
    tf.summary.image('I1', I1PH[:,:,:,0:3])
    tf.summary.image('I2', I2PH[:,:,:,0:3])
    tf.summary.image('Mask', LabelPH[:,:,:,0:1])
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
                IBatch, I1Batch, I2Batch, LabelBatch = GenerateBatch(TrainNames, TrainLabels, Factor, ImageSize, MiniBatchSize, BasePath, MaxFrameDiff)

                FeedDict = {ImgPH: IBatch, LabelPH: LabelBatch, I1PH: I1Batch, I2PH: I2Batch} 
                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
                
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

            # Print timing information every epoch
            PrintStatement = 'Epoch ' + str(Epochs) + ' completed out of ' + str(NumEpochs) + ' loss:' + str(EpochLoss)
            cprint(PrintStatement, 'yellow')
            print('Last Epoch took ' + str(TimeLastEpoch) +  ' secs, time taken till now, ' + str(TotalTimeElapsed) +\
                  ' estimated time to completion of this epoch is ' + str(EstimatedTimeToCompletion))
            print('------------------------------------------------')


    # Tensorboard
            #tf.scalar_summary("EpochLoss", EpochLoss)    


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
    Parser.add_argument('--BasePath', default='/media/nitin/Research/EVDodge/DatasetChethanEvents/DeblurredHomography', help='Base path of images, Default:/media/nitin/Research/EVDodge/processed')
    Parser.add_argument('--NumEpochs', type=int, default=200, help='Number of Epochs to Train for, Default:200')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=256, help='Size of the MiniBatch to use, Default:256')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--CheckPointPath', default='../CheckpointsSeg/', help='Path to save checkpoints, Default:../CheckpointsSeg/')
    Parser.add_argument('--LogsPath', default='/media/nitin/Research/EVDodge/LogsSeg/', help='Path to save Logs, Default:/media/nitin/Research/EVDodge/LogsSeg/')
    Parser.add_argument('--LR', type=float, default=1e-3, help='Learning Rate, Default: 1e-3')
    Parser.add_argument('--MaxFrameDiff', type=int, default=1, help='Maximum Frame difference to feed into network, Default: 1')
    
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    GPUDevice = Args.GPUDevice
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    LearningRate = Args.LR
    MaxFrameDiff = Args.MaxFrameDiff

    # Set GPUDevice
    tu.SetGPU(GPUDevice)

    
    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)

    
    # If LogsPath doesn't exist make the path
    if(not (os.path.isdir(LogsPath))):
       os.makedirs(LogsPath)


    # Setup all needed parameters including file reading
    TrainNames, TrainLabels, OptimizerParams, SaveCheckPoint, Factor, ImageSize, NumTrainSamples = SetupAll(BasePath, LearningRate)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 6), name='Input')
    I1PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 3), name='I1')
    I2PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 3), name='I2')
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 2), name='Label') # 2 classes
    
    TrainOperation(ImgPH, I1PH, I2PH, LabelPH, TrainNames, TrainLabels, NumTrainSamples, Factor, ImageSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, DivTrain, LatestFile, MaxFrameDiff, LogsPath, BasePath)
        
    
if __name__ == '__main__':
    main()
 
