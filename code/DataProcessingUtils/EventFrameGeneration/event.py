from PyQt5.QtWidgets import *
import numpy as np
# import Tkinter as tk
from Tkinter import *
from PIL import ImageTk, Image
import cv2
import sys
import time
from joblib import Parallel, delayed
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool
from joblib import Parallel, delayed
import os


class Events(object):
	"""
	Numpy Record Array with 
		x: event x coordinate
		y: event y coordinate 
		p: polarity
		ts: timestamp in microseconds
	"""
	
	def __init__(self, n_events, width=240, height=180):

		self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.float64)], shape=(n_events))
		self.width = width
		self.height = height
		self.size = self.data.ts.size
		self.x = np.matrix([self.data['x']])
		self.y = np.matrix([self.data['y']])
		self.old_xy = np.concatenate((self.x, self.y), axis=0)
		self.ts = np.matrix([self.data['ts']])
		# self.update_cloud(self.data)

	def get_frame(self,frame_data):

		# print(frame_data.size)
		frame = np.rec.array(None, dtype=[('value', np.float16),('valid', np.bool_)], shape=(self.height, self.width))
		frame.valid.fill(False)
		frame.value.fill(0.)
		# print(frame.size)

		for datum in np.nditer(frame_data, flags=['zerosize_ok']):
			# print(datum['y'])
			ts_val = datum['ts']
			f_data = frame[datum['y'], datum['x']]
			f_data.value += 1

		img = frame.value/20*255
		img = img.astype('uint8')
		# img = np.piecewise(img, [img <= 0, (img > 0) & (img < 255), img >= 255], [0, lambda x: x, 255])
		# cv2.normalize(img,img,0,255,cv2.NORM_L1)
		cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
		img = cv2.flip(img, 1)
		img = np.rot90(img)
		# cv2.imshow('img_f', img)
		# cv2.waitKey(0)
		return img

	def get_frame_positive(self,frame_data):

		# print(frame_data.size)
		frame = np.rec.array(None, dtype=[('value', np.float16),('valid', np.bool_)], shape=(self.height, self.width))
		frame.valid.fill(False)
		frame.value.fill(0.)
		# print(frame.size)

		for datum in np.nditer(frame_data, flags=['zerosize_ok']):
			# print(datum['y'])
			ts_val = datum['ts']
			f_data = frame[datum['y'], datum['x']]
			if(datum['p']== True):
				f_data.value += 1

		img = frame.value/20*255 #np.amax(frame.value)
		# print(np.amax(frame.value))
		# input('max value')
		img = img.astype('uint8')
		cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
		img = cv2.flip(img, 1)
		img = np.rot90(img)
		# cv2.imshow('img_pos', img)
		# cv2.waitKey(0)
		return img

	def get_frame_negative(self,frame_data):

		# print(frame_data.size)
		frame = np.rec.array(None, dtype=[('value', np.float16),('valid', np.bool_)], shape=(self.height, self.width))
		frame.valid.fill(False)
		frame.value.fill(0.)
		# print(frame.size)

		for datum in np.nditer(frame_data):
			# print(datum['y'])
			ts_val = datum['ts']
			f_data = frame[datum['y'], datum['x']]
			if(datum['p']== False):
				f_data.value += 1

		img = frame.value/20*255
		img = img.astype('uint8')
		# img = np.piecewise(img, [img <= 0, (img > 0) & (img < 255), img >= 255], [0, lambda x: x, 255])
		cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
		img = cv2.flip(img, 1)
		img = np.rot90(img)
		# cv2.imshow('img_neg', img)
		# cv2.waitKey(0)
		return img

	def get_time_frame(self,frame_data, time_width):

		# print(frame_data.size)
		frame = np.rec.array(None, dtype=[('value', np.uint16),('time', np.float16 ), ('valid', np.bool_)], shape=(self.height, self.width))
		frame.valid.fill(False)
		frame.value.fill(0)
		frame.time.fill(0.)
		# print(frame.size)

		for datum in np.nditer(frame_data):
			# print(datum['y'])
			ts_val = datum['ts']
			f_data = frame[datum['y'], datum['x']]
			f_data.time += np.float16(ts_val)
			f_data.value += 1

		img = np.divide(frame.time, frame.value, out=np.zeros_like(frame.time), where=frame.value!=0)
		img = img/time_width*255
		img = img.astype('uint8')
		cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
		
		img = cv2.flip(img, 1)
		img = np.rot90(img)
		# cv2.imshow('img_time', img)
		# cv2.waitKey(0)
		return img

	def get_projection_mat(self, dx, dy, dz, theta):

		print("inside get_projection", dx,dy,dz,theta)
		frame = np.rec.array(None, dtype=[('value', np.uint16)], shape=(self.height, self.width))
		frame.value.fill(0)

		dx = dx*1e-3
		dy = dy*1e-3
		dz = dz*1e-3
		#Project matrix 
		start = time.time()
		k = np.matrix([[dx, dy]])
		con_k = np.repeat(k.T, self.old_xy.size/2, axis=1)
		c, s = np.cos(theta), np.sin(theta)
		R = np.matrix([[c,-s], [s,c]])
		new = self.old_xy - np.multiply((self.ts),( con_k + (dz*np.dot(R, self.old_xy))))
		end = time.time()
		print("Projection time", end-start)

		#Converstion of 2D to 1D array
		i = np.array(new[0,:] + self.width * new[1,:])
		i.astype(int)
		u_ele, c_ele = np.unique(i.T,return_counts=True)
		u_c = np.asarray((u_ele, c_ele))
		print(u_c.shape, self.width, self.height)
		
		start = time.time()
		
		# inputs = range(new.size/2)

		# for i in inputs:
		# 	if((new[0,i] >= self.width) or (new[0,i]<0) or (new[1,i] >= self.height) or (new[1,i] < 0)):
		# 		continue
		# 	xy = frame[int(new[1,i]), int(new[0,i])]
		# 	xy.value += 1

		inputs = range(u_c.size/2)
		for i in inputs:
			x = int(u_c[0,i]%self.width)
			y = int(u_c[0,i]/self.width)

			if((x >= self.width) or (x<0) or (y >= self.height) or (y < 0)):
				continue
			xy = frame[y,x]
			xy.value = u_c[1,i]

		end = time.time()
		print("For loop time", end-start)
		img = frame.value * 10
		print(img.max())
		# cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
		img = img.astype('uint8')
		# cv2.normalize(img,img,0,255,cv2.NORM_L1)
		# cv2.imshow('img_p', img)
		# cv2.waitKey(0)
		return img

	def update_cloud(self, event_data):

		self.x = np.matrix([event_data['x']])
		self.y = np.matrix([event_data['y']])
		self.old_xy = np.concatenate((self.x, self.y), axis=0)
		self.ts = np.matrix([event_data['ts']])
	
	def display_init(self, x,y, z, theta, timestamp, time_width=1e4):

		print("inside display_init")
		# time_width = 1e5
		self.data.ts = (self.data.ts - self.data.ts[0])
		frame_start = self.data[0+timestamp].ts
		frame_end = (self.data[int(timestamp+time_width)].ts)
		frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
		self.update_cloud(frame_data)
		img = self.get_projection_mat(x, y, z,theta)

		return img

	def display_iter(self, x,y, z, theta, timestamp, time_width=1e4):
		print("inside display_iter")
		img = self.get_projection_mat(x, y, z, theta)

		return img

	def display_timechange(self, x,y, z, theta, timestamp, time_width=1e4):
		print("inside display_timechange")
		self.data.ts = (self.data.ts - self.data.ts[0])
		t_max = self.data.ts[-1]
		print("t_max and t_0", t_max, self.data.ts[0])
		frame_start = self.data[0+timestamp].ts
		frame_end = (self.data[int(timestamp+time_width)].ts)
		frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
		self.update_cloud(frame_data)
		img = self.get_projection_mat(x, y, z,theta)

		return img

	def generate_unsync_frame(self, time_width, path):

		print("inside unsync frame")
		path_p = os.path.join(path, 'pos')
		path_n = os.path.join(path, 'neg')
		path_ef = os.path.join(path, 'events')
		frame_start = self.data[0].ts
		frame_end = self.data[0].ts+time_width
		# print("frame start, frame_end", frame_start, frame_end)
		cnt = 0
		while(frame_end < self.data.ts[-1]):
			frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
			# print(frame_end)
			frame_data.ts = (frame_data.ts - frame_data.ts[0])
			img_p = self.get_frame_positive(frame_data)
			# cv2.imwrite(os.path.join(path_p, 'pos_'+str(int(frame_start*1e3/100))+'.jpg'), img_p)
			img_n = self.get_frame_negative(frame_data)
			# cv2.imwrite(os.path.join(path_n, 'neg_'+str(int(frame_start*1e3/100))+'.jpg'), img_n)
			# time = time+time_width
			img_t = self.get_time_frame(frame_data, time_width)
			frame_start = frame_end
			frame_end = frame_start+time_width
			# both = np.dstack((img_p,img_n))
			# null_frame = np.rec.array(None, dtype=[('value', np.uint16)], shape=(self.width, self.height))
			# null_frame.value.fill(0)
			# null_frame = null_frame.astype('uint8')
			event_frame = cv2.merge((img_p, img_n, img_t))
			cv2.imwrite(os.path.join(path_ef, 'event_'+str(cnt)+'.png'), event_frame)
			cnt = cnt+1
			# cv2.imwrite(os.path.join(str(int(frame_start*1e3/100))+'.jpg'), both)
			# cv2.imshow('img', both)
			# cv2.waitKey(0)

	def generate_unsync_avg_frame(self, time_width, path):

		print("inside unsync frame")
		path_p = os.path.join(path, 'pos')
		path_n = os.path.join(path, 'neg')
		path_ef = os.path.join(path, 'events')
		frame_start = self.data[0].ts
		frame_end = self.data[0].ts+time_width
		# print("frame start, frame_end", frame_start, frame_end)
		cnt = 0
		while(frame_end < self.data.ts[-1]):
			frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
			# print(frame_end)
			frame_data.ts = (frame_data.ts - frame_data.ts[0])
			img_ = self.get_frame(frame_data)
			# cv2.imwrite(os.path.join(path_p, 'pos_'+str(int(frame_start*1e3/100))+'.jpg'), img_p)
			# img_n = self.get_frame_negative(frame_data)
			# cv2.imwrite(os.path.join(path_n, 'neg_'+str(int(frame_start*1e3/100))+'.jpg'), img_n)
			# time = time+time_width
			img_t = self.get_time_frame(frame_data, time_width)
			frame_start = frame_end
			frame_end = frame_start+time_width
			# both = np.dstack((img_p,img_n))
			null_frame = np.rec.array(None, dtype=[('value', np.uint16)], shape=(self.width, self.height))
			null_frame.value.fill(0)
			null_frame = null_frame.astype('uint8')
			event_frame = cv2.merge((img_, null_frame, img_t))
			cv2.imwrite(os.path.join(path_ef, 'event_'+str(cnt)+'.png'), event_frame)
			cnt = cnt+1
			# cv2.imwrite(os.path.join(str(int(frame_start*1e3/100))+'.jpg'), both)
			# cv2.imshow('img', both)
			# cv2.waitKey(0)
	def generate_frame(self, time_width, path):

		print("inside display synced frame")
		path_p = os.path.join(path, 'pos')
		path_n = os.path.join(path, 'neg')
		path_ef = os.path.join(path, 'events')
		path_image = os.path.join(path, 'images.txt')
		# img_file = open(path_image,'r')
		with open(path_image) as fp:
			cnt = 0
			for line in fp:
				inner_list = [elt.strip() for elt in line.split(' ')]
				# print(inner_list[0])
				frame_start = float(inner_list[0])
				frame_end = float(inner_list[0])+time_width
				# print("frame start, frame_end", frame_start, frame_end)
				frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
				frame_data.ts = (frame_data.ts - frame_data.ts[0])
				img_p = self.get_frame_positive(frame_data)
			 	# cv2.imwrite(os.path.join(path_p, 'pos_'+str(cnt)+'.jpg'), img_p)
				img_n = self.get_frame_negative(frame_data)
				# cv2.imwrite(os.path.join(path_n, 'neg_'+str(cnt)+'.jpg'), img_n)
				null_frame = np.rec.array(None, dtype=[('value', np.uint16)], shape=(self.width, self.height))
				null_frame.value.fill(0)
				null_frame = null_frame.astype('uint8')
				event_frame = cv2.merge((img_p, img_n, null_frame))
				cv2.imwrite(os.path.join(path_ef, 'event_'+str(cnt)+'.png'), event_frame)
				cnt = cnt+1
				# cv2.imshow('img', event_frame)
				# cv2.waitKey(0)

	def generate_evframe(self, time_width, path):

		print("inside display synced frame")
		path_p = os.path.join(path, 'pos')
		path_n = os.path.join(path, 'neg')
		path_ef = os.path.join(path, 'events')
		path_image = os.path.join(path, 'images.txt')
		# img_file = open(path_image,'r')
		with open(path_image) as fp:
			cnt = 0
			for line in fp:
				inner_list = [elt.strip() for elt in line.split(' ')]
				# print(inner_list[0])
				frame_start = float(inner_list[0])
				frame_end = float(inner_list[0])+time_width
				# print("frame start, frame_end", frame_start, frame_end)
				frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
				frame_data.ts = (frame_data.ts - frame_data.ts[0])
				img_p = self.get_frame_positive(frame_data)
			 	# cv2.imwrite(os.path.join(path_p, 'pos_'+str(cnt)+'.jpg'), img_p)
				img_n = self.get_frame_negative(frame_data)
				# cv2.imwrite(os.path.join(path_n, 'neg_'+str(cnt)+'.jpg'), img_n)
				null_frame = np.rec.array(None, dtype=[('value', np.uint16)], shape=(self.width, self.height))
				null_frame.value.fill(0)
				null_frame = null_frame.astype('uint8')
				event_frame = cv2.merge((img_p, img_n, null_frame))
				cv2.imwrite(os.path.join(path_ef, 'event_'+str(cnt)+'.png'), event_frame)
				cnt = cnt+1
				cv2.imshow('img', event_frame)
				cv2.waitKey(0)

	def generate_evtime(self, time_width, path):

		print("inside display synced frame")
		path_p = os.path.join(path, 'pos')
		path_n = os.path.join(path, 'neg')
		path_ef = os.path.join(path, 'events')
		path_image = os.path.join(path, 'images.txt')
		# img_file = open(path_image,'r')
		with open(path_image) as fp:
			cnt = 0
			for line in fp:
				junk, frame_id = line.split('/') 
				cnt = int(frame_id[6:-5])
				inner_list = [elt.strip() for elt in line.split(' ')]
				# print(inner_list[0])
				frame_start = float(inner_list[0])
				frame_end = float(inner_list[0])+time_width
				# print("frame start, frame_end", frame_start, frame_end)
				frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
				frame_data.ts = (frame_data.ts - frame_data.ts[0])
				img_p = self.get_frame_positive(frame_data)
			 	# cv2.imwrite(os.path.join(path_p, 'pos_'+str(cnt)+'.jpg'), img_p)
				img_n = self.get_frame_negative(frame_data)
				# cv2.imwrite(os.path.join(path_n, 'neg_'+str(cnt)+'.jpg'), img_n)
				img_t = self.get_time_frame(frame_data, time_width)

				event_time_frame = cv2.merge((img_p, img_n, img_t))
				cv2.imwrite(os.path.join(path_ef, 'event_'+str(cnt)+'.png'), event_time_frame)
				# cnt = cnt+1
				# cv2.imshow('img', img_n)
				# cv2.waitKey(0)

	def generate_multithread_evtime(self, time_width, path):

		print("inside display synced frame")
		path_p = os.path.join(path, 'pos')
		path_n = os.path.join(path, 'neg')
		path_ef = os.path.join(path, 'events')
		# try: 
  #       	os.makedirs(path_ef)
  #   	except OSError:
  #       	if not os.path.isdir(path_ef):
  #           	raise
		path_image = os.path.join(path, 'images.txt')
		# img_file = open(path_image,'r')

		def evtime_image(line):
				junk, frame_id = line.split('/')
				cnt = 0
				cnt = int(frame_id[6:-4])
				inner_list = [elt.strip() for elt in line.split(' ')]
				# print(inner_list[0])
				frame_start = float(inner_list[0])
				frame_end = float(inner_list[0])+time_width
				# print("frame start, frame_end", frame_start, frame_end)
				frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
				frame_data.ts = (frame_data.ts - frame_data.ts[0])
				img_p = self.get_frame_positive(frame_data)
			 	# cv2.imwrite(os.path.join(path_p, 'pos_'+str(cnt)+'.jpg'), img_p)
				img_n = self.get_frame_negative(frame_data)
				# cv2.imwrite(os.path.join(path_n, 'neg_'+str(cnt)+'.jpg'), img_n)
				img_t = self.get_time_frame(frame_data, time_width)

				event_time_frame = cv2.merge((img_p, img_n, img_t))
				cv2.imwrite(os.path.join(path_ef, 'event_'+str(cnt)+'.png'), event_time_frame)	

		imagesList = [line.rstrip('\n') for line in open(path_image)]

		pool = ThreadPool(16)
		pool.map(evtime_image, imagesList)


	def generate_depth_evtime(self, time_width, path):

		print("inside generate_depth_evtime")
		path_p = os.path.join(path, 'pos')
		path_n = os.path.join(path, 'neg')
		path_ev = os.path.join(path, 'events')
		path_dp = os.path.join(path,'depths.txt')
		path_im = os.path.join(path, 'images.txt')

		with open(path_im) as fp:
			image_list = []
			for line in fp:
				image_list.append(line)
				# print(float(image_list[-1][0:20]))
		image_index = []
		for i in range (len(image_list)):
			image_index.append(float(image_list[i][0:20]) )

		# print(image_index)
		# np.asarray(image_index)
		# print(image_index.flat[np.abs(image_index - 1519939976.27).argmin()])
		# input('img')
		
		def find_nearest(array, value):
			array = np.asarray(array)
			idx = (np.abs(array - value)).argmin()
			return idx
		# print(find_nearest(image_index,1519939976.27))

		# def evtime_depth(line):
		# 		junk, frame_id = line.split('/')
		# 		cnt = 0
		# 		cnt = int(frame_id[6:-4])
		# 		inner_list = [elt.strip() for elt in line.split(' ')]
		# 		# print(inner_list[0])

		# 		# Image matching



		# 		# Event frame generation
		# 		frame_start = float(inner_list[0])
		# 		frame_end = float(inner_list[0])+time_width
		# 		# print("frame start, frame_end", frame_start, frame_end)
		# 		frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
		# 		frame_data.ts = (frame_data.ts - frame_data.ts[0])
		# 		img_p = self.get_frame_positive(frame_data)
		# 	 	# cv2.imwrite(os.path.join(path_p, 'pos_'+str(cnt)+'.jpg'), img_p)
		# 		img_n = self.get_frame_negative(frame_data)
		# 		# cv2.imwrite(os.path.join(path_n, 'neg_'+str(cnt)+'.jpg'), img_n)
		# 		img_t = self.get_time_frame(frame_data)

		# 		event_time_frame = cv2.merge((img_p, img_n, img_t))
		# 		cv2.imwrite(os.path.join(path_ev, 'event_'+str(cnt)+'.png'), event_time_frame)	

		# depthList = [line.rstrip('\n') for line in open(path_dp)]

		# pool = ThreadPool(16)
		# pool.map(evtime_depth, depthList)
		with open(os.path.join(path, 'depth_image.txt'), 'w') as di:
			with open(path_dp) as fp:
				for line in fp:
					junk, frame_id = line.split('/') 																							
					cnt = int(frame_id[6:-5])
					inner_list = [elt.strip() for elt in line.split(' ')]
					# print(inner_list[0])

					frame_start = float(inner_list[0])
					frame_end = float(inner_list[0])+time_width

					# Depth and image matching


					idx = find_nearest(image_index, frame_start)
					frame_list = [elt.strip() for elt in image_list[idx].split(' ')]
					print(inner_list[0], inner_list[1], frame_list[1])
					di.write('%s %s %s\n' % (inner_list[0], inner_list[1], frame_list[1]))


					# Event image generation
					# print("frame start, frame_end", frame_start, frame_end)
					frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
					frame_data.ts = (frame_data.ts - frame_data.ts[0])
					img_p = self.get_frame_positive(frame_data)
				 	# cv2.imwrite(os.path.join(path_p, 'pos_'+str(cnt)+'.jpg'), img_p)
					img_n = self.get_frame_negative(frame_data)
					# cv2.imwrite(os.path.join(path_n, 'neg_'+str(cnt)+'.jpg'), img_n)
					img_t = self.get_time_frame(frame_data, time_width)

					event_time_frame = cv2.merge((img_p, img_n, img_t))
					cv2.imwrite(os.path.join(path_ev, 'event_'+str(cnt)+'.png'), event_time_frame)
				# cnt = cnt+1
				# cv2.imshow('img', img_n)
				# cv2.waitKey(0)