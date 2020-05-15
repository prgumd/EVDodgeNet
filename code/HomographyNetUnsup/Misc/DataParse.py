#!/usr/bin/env python


# Dependencies:
# opencv, do (pip install opencv-python)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)

import glob
import os
from termcolor import colored, cprint
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import random

def MakeImgPairs(ReadPath, WritePath):
    DirNames = open(WritePath + os.sep + 'DirNames.txt', 'w')
    Labels = open(WritePath + os.sep + 'Labels.txt', 'w')
    for dirs in os.listdir(ReadPath):
        NumImgsStack = 2 # How many images to stack?
        for dir in tqdm(dirs):
            CurrReadPath = ReadPath + os.sep + dir + os.sep
            NumFilesInCurrDir = len(glob.glob(CurrReadPath + 'events' + os.sep + '*.png'))
            I = cv2.imread(CurrReadPath + 'events' + os.sep + "frame_%08d"%1 + '.png')
            ImageSize = np.shape(I)
            CurrWritePath = WritePath + os.sep + dir
            if(not os.path.exists(CurrWritePath + os.sep + 'events')):
                os.makedirs(CurrWritePath + os.sep + 'events')
            if(not os.path.exists(CurrWritePath + os.sep + 'label')):
                os.makedirs(CurrWritePath + os.sep + 'label')

            for ImgNum in tqdm(range(NumImgsStack, NumFilesInCurrDir+1)):
                for Img in range(NumImgsStack):
                    INow = cv2.imread(CurrReadPath + 'events' + os.sep + "frame_%08d"%(ImgNum-NumImgsStack+Img+1) + '.png')
                    MaskNow = np.array(cv2.imread(CurrReadPath + 'label' + os.sep + "label_%08d"%(ImgNum-NumImgsStack+Img+1) + '.png'))
                    MaskNow = MaskNow.astype(float)
                    if(Img==0):
                        IStacked = INow
                        MaskStacked = MaskNow
                    else:
                        IStacked = np.hstack((IStacked, INow))
                        MaskStacked = MaskStacked + MaskNow
                
                MaskStacked = MaskStacked/NumImgsStack
                _, MaskStacked = cv2.threshold(MaskStacked, 127, 255, cv2.THRESH_BINARY)
                cv2.imwrite(CurrWritePath + os.sep + 'events' + os.sep +  "frame_%08d"%(ImgNum-NumImgsStack+1) + '.png', IStacked)
                cv2.imwrite(CurrWritePath + os.sep + 'label' + os.sep +  "label_%08d"%(ImgNum-NumImgsStack+1) + '.png', MaskStacked)
                DirNames.write(CurrWritePath + os.sep + 'events' + os.sep  +  "frame_%08d"%(ImgNum-NumImgsStack+1) + '.png' + '\n')
                Labels.write(CurrWritePath + os.sep + 'label' + os.sep  +  "frame_%08d"%(ImgNum-NumImgsStack+1) + '.png' + '\n')
    DirNames.close()
    Labels.close()

def SetupSplits(Ratios, WritePath): 
    """
    Inputs: 
    Ratios: 3x1 list of ratio for Train, Validation and Test, each value lies between [0,1]
    Outputs:
    Writes 3 text files ./TxtFiles/Train.txt, ./TxtFiles/Val.txt and ./TxtFiles/Test.txt with Idxs corresponding to ./TxtFiles/DirNames.txt
    """
    # Ratios is a list of ratios from [0,1] for Training, Validation and Testing
    DirFiles = ReadDirNames(WritePath + os.sep + 'DirNames.txt')
    NumFiles = len(DirFiles)
    RandIdxs = range(NumFiles)
    random.shuffle(RandIdxs)
    TrainIdxs = RandIdxs[0:int(np.floor(NumFiles*Ratios[0]))]
    ValIdxs = RandIdxs[int(np.floor(NumFiles*Ratios[0])+1):int(np.floor(NumFiles*Ratios[0]+NumFiles*Ratios[1]))]
    TestIdxs = RandIdxs[int(np.floor(NumFiles*Ratios[0]+NumFiles*Ratios[1])):int(NumFiles-1)]

    if(not (os.path.isfile(WritePath + os.sep + 'Train.txt'))):
        Train = open(WritePath + os.sep + 'Train.txt', 'w')
        for TrainNum in range(len(TrainIdxs)):
            Train.write(str(TrainIdxs[TrainNum])+'\n')
        Train.close()
    else:
        cprint('WARNING: Train.txt File exists', 'yellow')

    if(not (os.path.isfile(WritePath + os.sep + 'Val.txt'))):
        Val = open(WritePath + os.sep + 'Val.txt', 'w')
        for ValNum in range(len(ValIdxs)):
            Val.write(str(ValIdxs[ValNum])+'\n')
        Val.close()
    else:
        cprint('WARNING: Val.txt File exists', 'yellow')

    if(not (os.path.isfile(WritePath + os.sep + 'Test.txt'))):
        Test = open(WritePath + os.sep + 'Test.txt', 'w')
        for TestNum in range(0, len(TestIdxs)):
            Test.write(str(TestIdxs[TestNum])+'\n')
        Test.close()
    else:
        cprint('WARNING: Test.txt File exists', 'yellow')
    # Read Splits once processed
    Train, Val, Test = ReadSplits(WritePath)

    return Train, Val, Test

def ReadDirNames(DirNamesPath):
    """
    Inputs: 
    None
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(DirNamesPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames

def ReadSplits(WritePath):
    """
    Inputs: 
    None
    Outputs:
    Train, Val and Test are data loaded from ./TxtFiles/Train.txt, ./TxtFiles/Val.txt and ./TxtFiles/Test.txt respectively
    They contain the Idxs of all the images to be used for training/validation (held-out testing in this case)/testing
    """
    Train = open(WritePath + os.sep + 'Train.txt', 'r')
    Train = Train.read()
    Train = map(int, Train.split())

    Val = open(WritePath + os.sep + 'Val.txt', 'r')
    Val = Val.read()
    Val = map(int, Val.split())

    Test = open(WritePath + os.sep + 'Test.txt', 'r')
    Test = Test.read()
    Test = map(int, Test.split())

    return Train, Val, Test

def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ReadPath', default='/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/dataset',
                        help='Base path of images, Default:/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/dataset')
    Parser.add_argument('--WritePath', default='/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/processed',
                        help='Base path of images, Default:/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/processed')
    Args = Parser.parse_args()
    ReadPath = Args.ReadPath
    WritePath = Args.WritePath
    
    if(not os.path.exists(WritePath)):
        cprint("WARNING: %s doesnt exist, Creating it."%WritePath, 'yellow')
        os.mkdir(WritePath)
        
    
    MakeImgPairs(ReadPath, WritePath)
    Ratios = [0.9, 0.00, 0.10]
    Train, Val, Test = SetupSplits(Ratios, WritePath)
    
if __name__ == '__main__':
    main()
