#!/usr/bin/python
"""
# ==============================
# Computes optical flow from two poses and depth images
# Author: Chethan Parameshwara
# Date: 20th May 2020
# ==============================
# Partially adapted from https://github.com/daniilidis-group/mvsec/tree/master/tools/gt_flow
"""

import numpy as np
import os
import cv2
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.linalg import logm
from flow import Flow 

 
def main():

    if os.path.isdir(sys.argv[1]):
    	fl = Flow(os.path.join(sys.argv[1], 'depthmaps.txt'), os.path.join(sys.argv[1], 'images.txt'), \
    		os.path.join(sys.argv[1], 'masks_1.txt'), os.path.join(sys.argv[1], 'masks_2.txt'), \
    		os.path.join(sys.argv[1], 'masks_3.txt'), os.path.join(sys.argv[1], 'camera_pose.txt'), \
    		os.path.join(sys.argv[1], 'obj1_pose.txt'), os.path.join(sys.argv[1], 'obj2_pose.txt'),\
    		os.path.join(sys.argv[1], 'obj2_pose.txt'), os.path.join(sys.argv[1], 'calib.txt'), \
    		os.path.join(sys.argv[1], 'events_imgs.txt'), os.path.join(sys.argv[1], 'flow_gt_1.txt'),
    		os.path.join(sys.argv[1], 'flow_pred.txt'),os.path.join(sys.argv[1], 'masks_full.txt'), [346,260], 4)
    	fl.full_flow()

if __name__ == "__main__":
    main()
