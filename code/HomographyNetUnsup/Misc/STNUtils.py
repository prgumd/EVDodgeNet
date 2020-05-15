#!/usr/bin/env python
# Adapted from Unsupervised Deep Homography: A Fast and Robust Homography Estimation Model 
# https://arxiv.org/abs/1709.03966
# https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018

import tensorflow as tf
import numpy as np
from Misc.TFSpatialTransformer import transformer
from Misc.AuxUtils import *


def solve_DLT(MiniBatchSize, AllPts, prHVal):

	batch_size = MiniBatchSize
	
	# Solve for H using DLT
	# pred_h4p_tile = tf.expand_dims(self.pred_h4p, [2]) # BATCH_SIZE x 8 x 1
	# AllPtsCol = np.reshape(AllPts, (np.product(AllPts.shape), 1))
	AllPtsCol = tf.reshape(AllPts, (-1, 8, 1)) 
	pts_1_tile = AllPtsCol
	prHValCol = tf.reshape(prHVal, (-1, 8, 1)) 
	# 4 points on the second image
	pred_pts_2_tile = tf.add(prHValCol, AllPtsCol)


	# Auxiliary tensors used to create Ax = b equation
	M1 = tf.constant(Aux_M1, tf.float32)
	M1_tensor = tf.expand_dims(M1, [0])
	M1_tile = tf.tile(M1_tensor,[batch_size,1,1])

	M2 = tf.constant(Aux_M2, tf.float32)
	M2_tensor = tf.expand_dims(M2, [0])
	M2_tile = tf.tile(M2_tensor,[batch_size,1,1])

	M3 = tf.constant(Aux_M3, tf.float32)
	M3_tensor = tf.expand_dims(M3, [0])
	M3_tile = tf.tile(M3_tensor,[batch_size,1,1])

	M4 = tf.constant(Aux_M4, tf.float32)
	M4_tensor = tf.expand_dims(M4, [0])
	M4_tile = tf.tile(M4_tensor,[batch_size,1,1])

	M5 = tf.constant(Aux_M5, tf.float32)
	M5_tensor = tf.expand_dims(M5, [0])
	M5_tile = tf.tile(M5_tensor,[batch_size,1,1])

	M6 = tf.constant(Aux_M6, tf.float32)
	M6_tensor = tf.expand_dims(M6, [0])
	M6_tile = tf.tile(M6_tensor,[batch_size,1,1])


	M71 = tf.constant(Aux_M71, tf.float32)
	M71_tensor = tf.expand_dims(M71, [0])
	M71_tile = tf.tile(M71_tensor,[batch_size,1,1])

	M72 = tf.constant(Aux_M72, tf.float32)
	M72_tensor = tf.expand_dims(M72, [0])
	M72_tile = tf.tile(M72_tensor,[batch_size,1,1])

	M8 = tf.constant(Aux_M8, tf.float32)
	M8_tensor = tf.expand_dims(M8, [0])
	M8_tile = tf.tile(M8_tensor,[batch_size,1,1])

	Mb = tf.constant(Aux_Mb, tf.float32)
	Mb_tensor = tf.expand_dims(Mb, [0])
	Mb_tile = tf.tile(Mb_tensor,[batch_size,1,1])

	# Form the equations Ax = b to compute H
	# Form A matrix
	A1 = tf.matmul(M1_tile, pts_1_tile) # Column 1
	A2 = tf.matmul(M2_tile, pts_1_tile) # Column 2
	A3 = M3_tile                   # Column 3
	A4 = tf.matmul(M4_tile, pts_1_tile) # Column 4
	A5 = tf.matmul(M5_tile, pts_1_tile) # Column 5
	A6 = M6_tile                   # Column 6
	A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, pts_1_tile)# Column 7
	A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, pts_1_tile)# Column 8

	A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
	                               tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
	                               tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
	     tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8
	# print('--Shape of A_mat:', A_mat.get_shape().as_list())
	# Form b matrix
	b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
	# print('--shape of b:', b_mat.get_shape().as_list())

	# Solve the Ax = b
	H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
	# print('--shape of H_8el', H_8el)


	# Add ones to the last cols to reconstruct H for computing reprojection error
	h_ones = tf.ones([batch_size, 1, 1])
	H_9el = tf.concat([H_8el,h_ones],1)
	H_flat = tf.reshape(H_9el, [-1,9])
	HMat = tf.reshape(H_flat,[-1,3,3])   # BATCH_SIZE x 3 x 3
	# H_flat = tf.reshape(HMat, [-1,9])
	# H_flatExcLastElem = H_flat[:, 0:8]

	return  HMat # H_flatExcLastElem 


def transform(ImageSize, HMat, MiniBatchSize, I1): 
	M = np.array([[ImageSize[1]/2.0, 0., ImageSize[1]/2.0],
	              [0., ImageSize[0]/2.0, ImageSize[0]/2.0],
	              [0., 0., 1.]]).astype(np.float32)
	M_tensor  = tf.constant(M, tf.float32)
	M_tile   = tf.tile(tf.expand_dims(M_tensor, [0]), [MiniBatchSize, 1,1])
	# Inverse of M
	M_inv = np.linalg.inv(M)
	M_tensor_inv = tf.constant(M_inv, tf.float32)
	M_tile_inv   = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [MiniBatchSize,1,1])
	# Transform H_mat since we scale image indices in transformer
	H_mat = tf.matmul(tf.matmul(M_tile_inv, HMat), M_tile)
	# H_flat = tf.reshape(H_mat, [-1,9])
	# H_flatExcLastElem = H_flat[:, 0:8]
	# Transform image 1 (large image) to image 2
	out_size = (ImageSize[0], ImageSize[1])
	# warped_image = tf.contrib.image.transform(I1, H_flatExcLastElem, interpolation='BILINEAR', output_shape=None, name=None)
	warped_image, _ = transformer(I1, H_mat, out_size)
	# TODO: warp image 2 to image 1

	# # width = ImageSize[1]; height = ImageSize[0]
 #    y_t = tf.range(0, MiniBatchSize*ImageSize[1]*ImageSize[0], ImageSize[1]*ImageSize[0]) 
 #    z =  tf.tile(tf.expand_dims(y_t,[1]),[1,PatchSize[1]*PatchSize[0]])
 #    batch_indices_tensor = tf.reshape(z, [-1]) # Add these value to patch_indices_batch[i] for i in range(num_pairs) # [BATCH_SIZE*WIDTH*HEIGHT]

	# # Extract the warped patch from warped_images by flatting the whole batch before using indices
	# # Note that input I  is  3 channels so we reduce to gray
	# warped_gray_images = tf.reduce_mean(warped_image, 3)
	# warped_images_flat = tf.reshape(warped_gray_images, [-1])
	# patch_indices_flat = tf.reshape(self.patch_indices, [-1])
	# pixel_indices =  patch_indices_flat + batch_indices_tensor
	# pred_I2_flat = tf.gather(warped_images_flat, pixel_indices)

	# warped_patch = tf.reshape(pred_I2_flat, [MiniBatchSize, PatchSize[1], PatchSize[0], 1])
	
	return warped_image 

# def get_mesh_grid_per_img(c_w, c_h, x_start=0, y_start=0):
#   '''Get 1D array of indices of pixels in the image of size c_h x c_w'''
# 	x_t = tf.matmul(tf.ones([c_h, 1]),
# 	        tf.transpose(\
# 	            tf.expand_dims(\
# 	                tf.linspace(tf.cast(x_start,'float32'), tf.cast(x_start+c_w-1,'float32'), c_w), 1), [1,0]))
# 	y_t = tf.matmul(tf.expand_dims(\
# 	                tf.linspace(tf.cast(y_start,'float32'), tf.cast(y_start+c_h-1,'float32'), c_h), 1),
# 	              tf.ones([1, c_w]))
# 	x_t_flat = tf.reshape(x_t, [-1]) # 1 x D
# 	y_t_flat = tf.reshape(y_t, [-1])

# 	return  x_t_flat, y_t_flat
