from PyQt5.QtWidgets import *
import event
import numpy as np
# import Tkinter as tk
from Tkinter import *
from PIL import ImageTk, Image
import cv2
import sys
import timeit

class GUI:
    
    def __init__(self, master=None, event_cloud=None):

    	# object.__init__()
    	self.w,self.h = event_cloud.width, event_cloud.height
    	self.gui_scale = 5;
    	self.img_scale = 3;
    	self.master = master
    	self.master.minsize(width=self.w*self.gui_scale, height=self.h*self.gui_scale)
    	self.master.maxsize(width=self.w*self.gui_scale, height=self.h*self.gui_scale)
    	self.label = Label(master, text="Event Annotator")
    	self.label.pack()

    	self.file = Button(master, text = 'Reset', command=self.reset)
    	self.event_cloud = event_cloud
    	cv_img = Image.fromarray(self.event_cloud.display_init(0,0,0,0,0))
    	cv_img = cv_img.resize((self.w*self.img_scale, self.h*self.img_scale), Image.ANTIALIAS)
    	self.image = ImageTk.PhotoImage(image = cv_img)
    	self.label = Label(image=self.image)
    	self.file.pack()
    	self.label.pack()

    	self.x = Scale(master, label ='x', from_=-10.0, to=10.0,length=600, resolution=0.1, orient=HORIZONTAL, command=self.display_iter)
    	self.x.set(0)
    	self.x.pack()
    	
    	self.y = Scale(master, label ='y', from_=-10.0, to=10.0,length=600, resolution=0.1, orient=HORIZONTAL, command=self.display_iter)
    	self.y.set(0)
    	self.y.pack()

    	self.z = Scale(master, label ='z', from_=-10.0, to=10.0,length=600, resolution=0.1, orient=HORIZONTAL, command=self.display_iter)
    	self.z.set(0)
    	self.z.pack()

    	self.theta = Scale(master, label ='theta', from_=-10.0, to=10.0,length=600, resolution=0.1, orient=HORIZONTAL, command=self.display_iter)
    	self.theta.set(0)
    	self.theta.pack()

    	self.time = Scale(master, label ='timestamp', from_=0, to=self.event_cloud.size, length=600, resolution=1e3, orient=HORIZONTAL, command=self.display_timechange)
    	self.time.set(0)
    	self.time.pack()

    	self.time_width = Scale(master, label ='time width', from_=0, to=1e6, length=600, resolution=1e4, orient=HORIZONTAL, command=self.display_timechange)
    	self.time_width.set(1e4)
    	self.time_width.pack()

    def display_init(self, event):
        # print (self.x.get(), self.y.get(), self.z.get(), self.theta.get())     
        cv_img =Image.fromarray(self.event_cloud.display_init(self.x.get(), self.y.get(), self.z.get(), self.theta.get(), self.time.get(), self.time_width.get()))
        cv_img = cv_img.resize((self.w*self.img_scale, self.h*self.img_scale), Image.ANTIALIAS)
        self.image= ImageTk.PhotoImage(image = cv_img)     
        self.label.configure(image = self.image)

    def display_iter(self, event):     
    	print (self.x.get(), self.y.get(), self.z.get(), self.theta.get())     
    	cv_img =Image.fromarray(self.event_cloud.display_iter(self.x.get(), self.y.get(), self.z.get(), self.theta.get(), self.time.get(), self.time_width.get()))
    	cv_img = cv_img.resize((self.w*self.img_scale, self.h*self.img_scale), Image.ANTIALIAS)
    	self.image= ImageTk.PhotoImage(image = cv_img)     
    	self.label.configure(image = self.image)

    def display_timechange(self, event):     
        print (self.x.get(), self.y.get(), self.z.get(), self.theta.get())     
        cv_img =Image.fromarray(self.event_cloud.display_timechange(self.x.get(), self.y.get(), self.z.get(), self.theta.get(), self.time.get(), self.time_width.get()))
        cv_img = cv_img.resize((self.w*self.img_scale, self.h*self.img_scale), Image.ANTIALIAS)
        self.image= ImageTk.PhotoImage(image = cv_img)     
        self.label.configure(image = self.image)

    def reset(self):
    	self.x.set(0)
    	self.y.set(0)
    	self.z.set(0)
    	self.theta.set(0)
        self.time.set(0)
        self.time_width.set(1e4)
    	# print (self.x.get(), self.y.get(), self.z.get(), self.theta.get())
    	cv_img = Image.fromarray(self.event_cloud.display_init(self.x.get(), self.y.get(), self.z.get(), self.theta.get(), self.time.get(), self.time_width.get()))
    	cv_img = cv_img.resize((self.w*self.img_scale, self.h*self.img_scale), Image.ANTIALIAS)
    	self.image= ImageTk.PhotoImage(image = cv_img)
    	self.label.configure(image = self.image)
