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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.linalg import logm
import OpenEXR
import Imath
import cv2
import tensorflow as tf
import PIL as pillow
from PIL import Image
from imageio import imread
from scipy import interpolate
import FlowUtils as fu
from tqdm import tqdm

try:
    from quaternion import quaternion
except ImportError:
    class quaternion:
        def __init__(self,w,x,y,z):
            self.w = w
            self.x = x
            self.y = y
            self.z = z
    
        def norm(self):
            return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
        def inverse(self):
            qnorm = self.norm()
            return quaternion(self.w/qnorm,
                              -self.x/qnorm,
                              -self.y/qnorm,
                              -self.z/qnorm)
    
        def __mul__(q1, q2):
            r = quaternion(q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z,
                           q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
                           q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
                           q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w)
            return r
    
        def __rmul__(q1, s):
            return quaternion(q1.w*s, q1.x*s, q1.y*s, q1.z*s)
    
        def __sub__(q1, q2):
            r = quaternion(q1.w-q2.w,
                           q1.x-q2.x,
                           q1.y-q2.y,
                           q1.z-q2.z)
            return r
    
        def __div__(q1, s):
            return quaternion(q1.w/s, q1.x/s, q1.y/s, q1.z/s)


class Flow:
    """
    - parameters
        - calibration :: a Calibration object from calibration.py
    """
    def __init__(self, depthfile, imagefile, maskfile1,  maskfile2,  maskfile3, \
                cameraposefile, objposefile1, objposefile2, objposefile3,  calibfile, \
                event_img, flow_gt, flow_pred, masks_full, resolution, step_size):

        # reading txt files
        # print("reading .... %s " % (cameraposefile))
        f = open(cameraposefile, 'rb')
        camera_pose_data = np.loadtxt(f)
        f.close()

        all_camera_pose_data = camera_pose_data

        # print("reading .... %s " % (objposefile1))
        f = open(objposefile1, 'rb')
        obj_pose_data1 = np.loadtxt(f)
        f.close()
        
        all_obj_pose_data_1 = obj_pose_data1

        # print("reading .... %s " % (objposefile2))
        f = open(objposefile2, 'rb')
        obj_pose_data2 = np.loadtxt(f)
        f.close()
        
        all_obj_pose_data_2 = obj_pose_data2

        # print("reading .... %s " % (objposefile3))
        f = open(objposefile3, 'rb')
        obj_pose_data3 = np.loadtxt(f)
        f.close()
        
        all_obj_pose_data_3 = obj_pose_data3

        # print("reading .... %s " % (depthfile))
        file = open(depthfile, 'r') 
        all_depth_data = []
        for line in file: 
            depth_data = []
            for word in line.split():
                depth_data.append(word)
            all_depth_data.append(depth_data)

        # print("reading .... %s " % (imagefile))
        file = open(imagefile, 'r') 
        all_image_data = []
        for line in file: 
            image_data = []
            for word in line.split():
                image_data.append(word)
            all_image_data.append(image_data)

        # print("reading .... %s " % (maskfile1))
        file = open(maskfile1, 'r') 
        all_mask_data_1 = []
        for line in file: 
            mask_data1 = []
            for word in line.split():
                mask_data1.append(word)
            all_mask_data_1.append(mask_data1)

        # print("reading .... %s " % (maskfile2))
        file = open(maskfile2, 'r') 
        all_mask_data_2 = []
        for line in file: 
            mask_data2 = []
            for word in line.split():
                mask_data2.append(word)
            all_mask_data_2.append(mask_data2)


        # print("reading .... %s " % (maskfile3))
        file = open(maskfile3, 'r') 
        all_mask_data_3 = []
        for line in file: 
            mask_data3 = []
            for word in line.split():
                mask_data3.append(word)
            all_mask_data_3.append(mask_data3)

        # print(all_pose_data[0][1])

        file = open(calibfile, 'r')
        # print("reading .... %s " % (calibfile)) 
        f = open(calibfile, 'rb')
        calib_data = np.loadtxt(f)
        f.close()
    
        # print("reading .... %s " % (event_img))
        file = open(event_img, 'r') 
        all_event_imgs = []
        for line in file: 
            event_data = []
            for word in line.split():
                event_data.append(word)
            all_event_imgs.append(event_data)

        # print("reading .... %s " % (flow_gt))
        file = open(flow_gt, 'r') 
        all_flow_gt = []
        for line in file: 
            flow_gt = []
            for word in line.split():
                flow_gt.append(word)
            all_flow_gt.append(flow_gt)


        # print("reading .... %s " % (flow_pred))
        file = open(flow_pred, 'r') 
        all_flow_pred = []
        for line in file: 
            flow_pred = []
            for word in line.split():
                flow_pred.append(word)
            all_flow_pred.append(flow_pred)

        # print("reading .... %s " % (masks_full))
        file = open(masks_full, 'r') 
        all_masks_full = []
        for line in file: 
            masks_full = []
            for word in line.split():
                masks_full.append(word)
            all_masks_full.append(masks_full)

    	# camera pose data 
    	self.pose_data = all_camera_pose_data
        self.obj_pose_1 = all_obj_pose_data_1
        self.obj_pose_2 = all_obj_pose_data_2
        self.obj_pose_3 = all_obj_pose_data_3
    	
        # depth
    	self.depth_data = all_depth_data
    	# print(self.depth_data[1][0])
        
        # image
        self.image_data = all_image_data
    	
        # mask data
        self.mask_data_1 = all_mask_data_1
        self.mask_data_2 = all_mask_data_2
        self.mask_data_3 = all_mask_data_3
        self.masks_full = all_masks_full


        # image information
        self.resolution = resolution
        self.step_size = step_size

        # event images for evaluation
        self.event_img = all_event_imgs

        # flow files for evaluation
        self.flow_gt = all_flow_gt
        self.flow_pred = all_flow_pred


        # average image count

        self.avg_image_count = 0

        # visualization

        self.visualization = False

        # avg number of objects 

        self.f = open('avg_obj_count.txt','w')

    	# calibration data

    	self.K = np.array([[calib_data[0], 0., calib_data[2]],
                           [0., calib_data[1], calib_data[3]],
                           [0., 0., 1.]])

    	x_inds, y_inds = np.meshgrid(np.arange(resolution[0]),
                                     np.arange(resolution[1]))
        x_inds = x_inds.astype(np.float32)
        y_inds = y_inds.astype(np.float32)

        x_inds -= self.K[0,2]
        x_inds *= (1./self.K[0,0])

        y_inds -= self.K[1,2]
        y_inds *= (1./self.K[1,1])

        self.flat_x_map = x_inds.reshape((-1))
        self.flat_y_map = y_inds.reshape((-1))

        N = self.flat_x_map.shape[0]

        self.omega_mat = np.zeros((N,2,3))

        self.omega_mat[:,0,0] = self.flat_x_map * self.flat_y_map
        self.omega_mat[:,1,0] = 1+ np.square(self.flat_y_map)

        self.omega_mat[:,0,1] = -(1+np.square(self.flat_x_map))
        self.omega_mat[:,1,1] = -(self.flat_x_map*self.flat_y_map)

        self.omega_mat[:,0,2] = self.flat_y_map
        self.omega_mat[:,1,2] = -self.flat_x_map
        self.hsv_buffer = None


    def p_q_t(self, pose):
        # converts ROS pose msg to numpy array
        p = np.array([pose[1], pose[2], pose[3]])
        q = quaternion(pose[7], pose[4], pose[5], pose[6])
        t = pose[0]
        return p, q, t


    def compute_velocity_from_msg(self, P0, P1):
        p0, q0, t0 = self.p_q_t(P0)
        p1, q1, t1 = self.p_q_t(P1)

        # There's something wrong with the current function to go from quat to matrix.
        # Using the TF version instead.
        q0_ros = [q0.x, q0.y, q0.z, q0.w]
        q1_ros = [q1.x, q1.y, q1.z, q1.w]
        
        import tf
        H0 = tf.transformations.quaternion_matrix(q0_ros)
        H0[:3, 3] = p0

        H1 = tf.transformations.quaternion_matrix(q1_ros)
        H1[:3, 3] = p1

        # Let the homogeneous matrix handle the inversion etc. Guaranteed correctness.
        H01 = np.dot(np.linalg.inv(H0), H1)
        dt = t1 - t0

        V = H01[:3, 3] / dt
        w_hat = logm(H01[:3, :3]) / dt
        Omega = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

        return V, Omega, dt


    def compute_obj_velocity_from_msg(self, C0, C1, O0, O1):
        cp0, cq0, ct0 = self.p_q_t(C0)
        cp1, cq1, ct1 = self.p_q_t(C1)

        op0, oq0, t0 = self.p_q_t(O0)
        op1, oq1, t1 = self.p_q_t(O1)


        # There's something wrong with the current function to go from quat to matrix.
        # Using the TF version instead.
        import tf
        cq0_ros = [cq0.x, cq0.y, cq0.z, cq0.w]
        cq1_ros = [cq1.x, cq1.y, cq1.z, cq1.w]

        oq0_ros = [oq0.x, oq0.y, oq0.z, oq0.w]
        oq1_ros = [oq1.x, oq1.y, oq1.z, oq1.w]
        

        CH0 = tf.transformations.quaternion_matrix(cq0_ros)
        CH0[:3, 3] = cp0

        CH1 = tf.transformations.quaternion_matrix(cq1_ros)
        CH1[:3, 3] = cp1

        OH0 = tf.transformations.quaternion_matrix(oq0_ros)
        OH0[:3, 3] = op0

        OH1 = tf.transformations.quaternion_matrix(oq1_ros)
        OH1[:3, 3] = op1

        
        # World to Camera frame 

        H0 = np.dot(np.linalg.inv(OH0), CH0)
        H1 = np.dot(np.linalg.inv(OH1), CH1)

        # Let the homogeneous matrix handle the inversion etc. Guaranteed correctness.
        H01 = np.dot(np.linalg.inv(H0), H1)
        dt = t1 - t0

        V = H01[:3, 3] / dt
        w_hat = logm(H01[:3, :3]) / dt
        Omega = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

        return V, Omega, dt


    def compute_flow_single_frame(self, V, Omega, depth_image, dt):
        """
        params:
            V : [3,1]
            Omega : [3,1]
            depth_image : [m,n]
        """
        flat_depth = depth_image.ravel()
        flat_depth[np.logical_or(np.isclose(flat_depth,0.0), flat_depth<0.)]

        mask = np.isfinite(flat_depth)

        fdm = 1./flat_depth[mask]
        fxm = self.flat_x_map[mask]
        fym = self.flat_y_map[mask]
        omm = self.omega_mat[mask,:,:]

        x_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_x_flow_out = x_flow_out.reshape((-1))
        flat_x_flow_out[mask] = fdm * (fxm*V[2]-V[0])
        flat_x_flow_out[mask] +=  np.squeeze(np.dot(omm[:,0,:], Omega))

        y_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_y_flow_out = y_flow_out.reshape((-1))
        flat_y_flow_out[mask] = fdm * (fym*V[2]-V[1])
        flat_y_flow_out[mask] +=  np.squeeze(np.dot(omm[:,1,:], Omega))

        flat_x_flow_out *= dt * self.K[0,0]
        flat_y_flow_out *= dt * self.K[1,1]

        return x_flow_out, y_flow_out


    def compute_flow_mask_frame(self, V, Omega, depth_image, dt):
        """
        params:
            V : [3,1]
            Omega : [3,1]
            depth_image : [m,n]
        """
        flat_depth = depth_image.ravel()
        flat_depth[np.logical_or(np.isclose(flat_depth,0.0), flat_depth<0.)]
        mask = flat_depth != 0.0
        fdm = 1./flat_depth[mask]
        fxm = self.flat_x_map[mask]
        fym = self.flat_y_map[mask]
        omm = self.omega_mat[mask,:,:]

        x_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_x_flow_out = x_flow_out.reshape((-1))
        flat_x_flow_out[mask] = fdm * (fxm*V[2]-V[0])
        flat_x_flow_out[mask] +=  np.squeeze(np.dot(omm[:,0,:], Omega))

        y_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_y_flow_out = y_flow_out.reshape((-1))
        flat_y_flow_out[mask] = fdm * (fym*V[2]-V[1])
        flat_y_flow_out[mask] +=  np.squeeze(np.dot(omm[:,1,:], Omega))

        flat_x_flow_out *= dt * self.K[0,0]
        flat_y_flow_out *= dt * self.K[1,1]

        return x_flow_out, y_flow_out


    def colorize_image(self, flow_x, flow_y):
        if self.hsv_buffer is None:
            self.hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1],3))
            self.hsv_buffer[:,:,1] = 1.0
        self.hsv_buffer[:,:,0] = (np.arctan2(flow_y,flow_x)+np.pi)/(2.0*np.pi)

        self.hsv_buffer[:,:,2] = np.linalg.norm( np.stack((flow_x,flow_y), axis=0), axis=0 )

        # self.hsv_buffer[:,:,2] = np.log(1.+self.hsv_buffer[:,:,2]) # hopefully better overall dynamic range in final video

        flat = self.hsv_buffer[:,:,2].reshape((-1))
        m = np.nanmax(flat[np.isfinite(flat)])
        if not np.isclose(m, 0.0):
            self.hsv_buffer[:,:,2] /= m

        return colors.hsv_to_rgb(self.hsv_buffer)

    def visualize_flow(self, flow_x, flow_y, fig):
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow( self.colorize_image(flow_x, flow_y) )

    def warp_image(self,im, flow):
        """
        Use optical flow to warp image to the next
        :param im: image to warp
        :param flow: optical flow
        :return: warped image
        """
        image_height = im.shape[0]
        image_width = im.shape[1]
        flow_height = flow.shape[0]
        flow_width = flow.shape[1]
        n = image_height * image_width
        (iy, ix) = np.mgrid[0:image_height, 0:image_width]
        (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
        fx = fx.astype(np.float64)
        fy = fy.astype(np.float64)
        fx += flow[:,:,0]
        fy += flow[:,:,1]
        mask = np.logical_or(fx <0 , fx > flow_width)
        mask = np.logical_or(mask, fy < 0)
        mask = np.logical_or(mask, fy > flow_height)
        fx = np.minimum(np.maximum(fx, 0), flow_width)
        fy = np.minimum(np.maximum(fy, 0), flow_height)
        points = np.concatenate((ix.reshape(n,1), iy.reshape(n,1)), axis=1)
        xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n,1)), axis=1)
        warp = np.zeros((image_height, image_width, im.shape[2]))
        for i in range(im.shape[2]):
            channel = im[:, :, i]
            values = channel.reshape(n, 1)
            new_channel = interpolate.griddata(points, values, xi, method='cubic')
            new_channel = np.reshape(new_channel, [flow_height, flow_width])
            new_channel[mask] = 1
            warp[:, :, i] = new_channel.astype(np.uint8)

        return warp.astype(np.uint8)

    def warp_flow(self,img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res

    def draw_flow(self,img, flow, step=1):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        # vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv2.polylines(img, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
        return img
    	    
    def rigid_flow(self):

    	V, Omega, dt = self.compute_velocity_from_msg(self.pose_data[200,:], self.pose_data[204,:])

    	# print(self.depth_data[1][1])

    	depth_image0 = OpenEXR.InputFile(self.depth_data[200][1])
    	dw0 = depth_image0.header()['dataWindow']
    	size0 = (dw0.max.x - dw0.min.x + 1, dw0.max.y - dw0.min.y + 1)
    	pt0 = Imath.PixelType(Imath.PixelType.FLOAT)
    	depth0 = np.fromstring(depth_image0.channel("Z"), dtype=np.float32)
        depth0.shape = (size0[1], size0[0])  # Numpy arrays are (row, col)

        depth_image1 = OpenEXR.InputFile(self.depth_data[204][1])
        dw1 = depth_image1.header()['dataWindow']
        size1 = (dw1.max.x - dw1.min.x + 1, dw1.max.y - dw1.min.y + 1)
        pt1 = Imath.PixelType(Imath.PixelType.FLOAT)
        depth1 = np.fromstring(depth_image1.channel("Z"), dtype=np.float32)
        depth1.shape = (size1[1], size1[0])  # Numpy arrays are (row, col)

        depth = (depth0+depth1)/2

    	flow_x_dist, flow_y_dist = self.compute_flow_single_frame(V,
                                                                  Omega,
                                                                  depth,
                                                                  dt)
    	print(flow_x_dist, flow_y_dist)

    	flow = np.dstack((flow_x_dist, flow_y_dist))
        flow = np.float32(flow)
    	# verfication 
    	img1 = cv2.imread(self.image_data[200][1],1)
        # img1 = np.float32(img1)
        print(img1.shape)
    	img2 = cv2.imread(self.image_data[204][1], 1)
        # img2 = np.float32(img2)
        print(img1.shape, flow.dtype)

    	warpped_img1 = self.warp_image(img2, flow)
        # warpped_img1 = self.warp_flow(img2, flow)

        cv2.imshow('warpped_img1', cv2.subtract(img1, warpped_img1))


        first_img = self.colorize_image(flow_x_dist, flow_y_dist)
        cv2.imshow('image',first_img)
        cv2.waitKey(0)


    def write_flow(self, flow, filename):
        """
        write optical flow in Middlebury .flo format
        :param flow: optical flow map
        :param filename: optical flow file path to be saved
        :return: None
        """
        f = open(filename, 'wb')
        magic = np.array([202021.25], dtype=np.float32)
        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        magic.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)
        f.close()


    def full_flow(self):

        for n in range(10000-self.step_size):

            end = n + self.step_size
            start = n
            V, Omega, dt = self.compute_velocity_from_msg(self.pose_data[start,:], self.pose_data[end,:])

        
            # print(self.depth_data[1][1])

            depth_image1 = OpenEXR.InputFile(self.depth_data[start][1])
            dw1 = depth_image1.header()['dataWindow']
            size1 = (dw1.max.x - dw1.min.x + 1, dw1.max.y - dw1.min.y + 1)
            pt1 = Imath.PixelType(Imath.PixelType.FLOAT)
            depth1 = np.fromstring(depth_image1.channel("Z"), dtype=np.float32)
            depth1.shape = (size1[1], size1[0])  # Numpy arrays are (row, col)

            depth_image2 = OpenEXR.InputFile(self.depth_data[end][1])
            dw2 = depth_image2.header()['dataWindow']
            size2 = (dw2.max.x - dw2.min.x + 1, dw2.max.y - dw2.min.y + 1)
            pt2 = Imath.PixelType(Imath.PixelType.FLOAT)
            depth2 = np.fromstring(depth_image2.channel("Z"), dtype=np.float32)
            depth2.shape = (size2[1], size2[0])  # Numpy arrays are (row, col)

            depth = (depth1+depth2)/2


            #rigid flow
            flow_x_dist, flow_y_dist = self.compute_flow_single_frame(V,Omega,depth,dt)

            flow = np.dstack((flow_x_dist, flow_y_dist))
            flow = np.float32(flow)

            # verfication 
            img1 = cv2.imread(self.image_data[start][1],1)
            img2 = cv2.imread(self.image_data[end][1],1)
            # warpped_img1 = self.warp_image(img2, flow)
            # cv2.imshow('warpped_img1', cv2.subtract(img1, warpped_img1))

            #non-rigid flow
            mask11 = np.float32(cv2.imread(self.mask_data_1[start][1],0))
            mask12 = np.float32(cv2.imread(self.mask_data_1[end][1],0))

            if(np.amax(mask11) > 0.0):
                print('mask11')

                V_obj, Omega_obj, dt_obj = self.compute_obj_velocity_from_msg(self.pose_data[start,:], self.pose_data[end,:],\
                                                                        self.obj_pose_1[start,:], self.obj_pose_1[end,:])

                flow_x_dist_obj, flow_y_dist_obj = self.compute_flow_mask_frame(V_obj,Omega_obj,depth,dt_obj)
                flow_mask = np.dstack((flow_x_dist_obj, flow_y_dist_obj))
                flow_mask = np.float32(flow_mask)

                idx11=(mask11==255)
                flow_x_dist[idx11]=flow_x_dist_obj[idx11]
                flow_y_dist[idx11]=flow_y_dist_obj[idx11]

                self.avg_image_count+= 1

            mask21 = np.float32(cv2.imread(self.mask_data_2[start][1],0))
            mask22 = np.float32(cv2.imread(self.mask_data_2[end][1],0))

            if(np.amax(mask21) > 0.0):
                print('mask21')

                V_obj, Omega_obj, dt_obj = self.compute_obj_velocity_from_msg(self.pose_data[start,:], self.pose_data[end,:],\
                                                                        self.obj_pose_2[start,:], self.obj_pose_2[end,:])

                flow_x_dist_obj, flow_y_dist_obj = self.compute_flow_mask_frame(V_obj,Omega_obj,depth,dt_obj)
                flow_mask = np.dstack((flow_x_dist_obj, flow_y_dist_obj))
                flow_mask = np.float32(flow_mask)

                idx21=(mask21==255)
                flow_x_dist[idx21]=flow_x_dist_obj[idx21]
                flow_y_dist[idx21]=flow_y_dist_obj[idx21]

                self.avg_image_count+= 1

            mask31 = np.float32(cv2.imread(self.mask_data_3[start][1],0))
            mask32 = np.float32(cv2.imread(self.mask_data_3[end][1],0))
            
            if(np.amax(mask31) > 0.0):
                print('mask31')

                V_obj, Omega_obj, dt_obj = self.compute_obj_velocity_from_msg(self.pose_data[start,:], self.pose_data[end,:],\
                                                                        self.obj_pose_3[start,:], self.obj_pose_3[end,:])

                flow_x_dist_obj, flow_y_dist_obj = self.compute_flow_mask_frame(V_obj,Omega_obj,depth,dt_obj)
                flow_mask = np.dstack((flow_x_dist_obj, flow_y_dist_obj))
                flow_mask = np.float32(flow_mask)

                idx31=(mask31==255)
                flow_x_dist[idx31]=flow_x_dist_obj[idx31]
                flow_y_dist[idx31]=flow_y_dist_obj[idx31]

                self.avg_image_count+= 1

            flow_combined = np.dstack((flow_x_dist, flow_y_dist))
            flow_combined = np.float32(flow_combined)

            self.write_flow(flow_combined, "./flow_gt/frame_%d_%d"%(end, start) + '.flo')

            # warpped_img2 = self.warp_image(img2, flow_combined)
            # print(cv2.sum(cv2.subtract(img1, warpped_img2)))
            # input('a')
            # cv2.imwrite("./flow_warped_images_1/flow_%d_%d"%(end, start) + ".jpg", cv2.subtract(img1, warpped_img2))

            if(self.visualization == True):
                warpped_img2 = self.warp_image(img2, flow_combined)

                cv2.imshow('warped_img2', cv2.subtract(img1, warpped_img2))
                cv2.waitKey(0)

    # def avg_object_count(self):

    #     for n in range(10000):

    #         mask11 = np.float32(cv2.imread(self.mask_data_1[n][1],0))

    #         if(np.amax(mask11) > 0.0):

    #             self.avg_image_count+= 1

    #         mask21 = np.float32(cv2.imread(self.mask_data_2[n][1],0))

    #         if(np.amax(mask21) > 0.0):

    #             self.avg_image_count+= 1

    #         mask31 = np.float32(cv2.imread(self.mask_data_3[n][1],0))
            
    #         if(np.amax(mask31) > 0.0):

    #             self.avg_image_count+= 1

            
    #         self.f.write(self.mask_data_1[n][0] + " " + str(self.avg_image_count) + "\n")

    #         self.avg_image_count = 0


    # def flow_error_dense(self, flow_gt, flow_pred, event_img, mask=None):
    #     # Only compute error over points that are valid in the GT (not inf or 0).

    #     # print(flow_gt.dtype)
    #     # print(flow_pred.dtype)
    #     # img = fu.flow_to_image(flow_gt)
    #     # img2 = fu.flow_to_image(flow_pred)
    #     # cv2.imshow('a', img)
    #     # cv2.imshow('b', img2)
    #     # cv2.waitKey(0)


    #     event_mask = event_img > 0
    #     flow_mask = np.logical_and(
    #         np.logical_and(~np.isinf(flow_pred[:, :, 0]), ~np.isinf(flow_pred[:, :, 1])),
    #         np.linalg.norm(flow_pred, axis=2) > 0)
    #     # print(event_mask.shape, flow_mask.shape, mask.shape )
    #     if (mask is not None):
    #         total_mask = np.squeeze(np.logical_and(np.logical_and(event_mask, flow_mask), mask))
    #     else:
    #         total_mask = np.squeeze(np.logical_and(event_mask, flow_mask))
    #     total_mask1 = total_mask.astype(np.uint8)  #convert to an unsigned byte
    #     total_mask1*=255
    #     # cv2.imshow('c', total_mask1)
        

    #     gt_masked = flow_gt[total_mask, :]
    #     pred_masked = flow_pred[total_mask, :]

    #     # Average endpoint error
    #     A = gt_masked - pred_masked
    #     if(np.shape(A)[0] == 0):
    #         AEE = 0.
    #         AAE = 0.
    #         percent_AEE = 0.
    #         percent_AAE = 0.
    #         n_points = 0
    #     else:
    #         EE = gt_masked - pred_masked
    #         EE = np.sqrt(EE[:, 0]**2 + EE[:, 1]**2)
    #         n_points = EE.shape[0]
    #         AEE = np.mean(EE)

    #         # Average Angle Error
    #         Num = np.sum(np.multiply(gt_masked, pred_masked), axis=1)
    #         Den = np.multiply(np.sqrt(gt_masked[:, 0]**2 + gt_masked[:, 1]**2), np.sqrt(pred_masked[:, 0]**2 + pred_masked[:, 1]**2))
    #         # LargeValsMask = np.sqrt(pred_masked[:, 0]**2 + pred_masked[:, 1]**2) > 5.0
    #         # AE = np.arccos(np.divide(Num[LargeValsMask], Den[LargeValsMask]))
    #         AE = np.arccos(np.divide(Num, Den))
    #         # print(gt_masked - pred_masked)
    #         # input('a')  
    #         AENan = np.isnan(AE)
    #         # Extract Non Nan numbers!
    #         AE = AE[np.logical_not(AENan)]
    #         AAE = np.mean(AE)

    #         # Percentage of points with EE < 3 pixels.
    #         eps = 5.
    #         tol = 0.25
    #         gt_maskedMag = np.sqrt(gt_masked[:, 0]**2 + gt_masked[:, 1]**2)
    #         tolEE = np.multiply(gt_maskedMag, tol)
    #         tolFinal = np.minimum(tolEE, eps)
    #         EEInlier = (EE <= tolFinal)
    #         # print(tolEE)
    #         # print(tolFinal)
    #         # print(EEInlier)
    #         # print(np.sum(EEInlier))
    #         # print(np.sum(EEInlier)/n_points*100.0)
    #         # print(AAE)
    #         # input('a')
    #         # print('AEE', AEE)
    #         # print('Max flow value', np.amax(flow_pred))
    #         percent_AEE = []
    #         percent_AAE = []

    #     return AEE, AAE, percent_AEE, percent_AAE, n_points


    # def flow_evaluation(self):

    #     AEE_total = 0

    #     for n in range(4998):
    #         # n = n+2000
    #         d = 0
    #         flow_gt = fu.read_flow(self.flow_gt[n][0])
    #         flow_pred = fu.read_flow(self.flow_pred[n-d][0])
    #         event_img = cv2.imread(self.event_img[n][0], 0)
    #         mask = cv2.imread(self.masks_full[n][1], 0)
    #         # mask = 255 - mask
    #         # input('a')
    #         AEE, AAE, percent_AEE, percent_AAE, n_points = self.flow_error_dense(flow_gt, flow_pred, event_img, mask=None)
    #         AEE_total = AEE_total + AEE
    #         print(n, AEE, AEE_total)

    #     AEE_avg = AEE_total/4997
    #     print(AEE_total, AEE_avg)
    #     # n = 4782
    #     # flow_gt = fu.read_flow(self.flow_gt[n][0])
    #     # flow_pred = fu.read_flow(self.flow_pred[n][0])
    #     # event_img = cv2.imread(self.event_img[n][0], 0)
    #     # mask = cv2.imread(self.masks_full[n][1], 0)
    #     # AEE, AAE, percent_AEE, percent_AAE, n_points = self.flow_error_dense(flow_gt, flow_pred, event_img, mask)
    #     # print(n, AEE, AAE, percent_AEE, percent_AAE, n_points)

    # def flow_view(self):

    #     flow_gt = fu.read_flow(self.flow_gt[1853][0])
    #     print(flow_gt.shape)
    #     input('a')

    #     first_img = self.colorize_image(flow_x_dist, flow_y_dist)
    #     cv2.imshow('image',first_img)
    #     cv2.waitKey(0)



