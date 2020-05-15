from gui import GUI
from event import Events  
from Tkinter import *
import numpy as np
from PIL import ImageTk, Image
import os
import cv2
import sys
import timeit

def read_eventfile(filename):
	
	print("reading .... %s " % (filename))
	f = open(filename, 'rb')
	raw_data = np.loadtxt(f)
	f.close()
	# print(raw_data.size)

	all_y = raw_data[:, 1]
	all_x = raw_data[:, 2]
	all_p = raw_data[:,3]
	all_ts = raw_data[:,0]
	# print(all_ts.size)
	td = Events(all_ts.size)
	td.data.x = all_x
	td.width = td.data.x.max() + 1
	td.data.y = all_y
	td.height = td.data.y.max() + 1
	td.data.ts = all_ts
	td.data.p = all_p
	print("event file read")
	return td

# def main():
#     root = Tk()
#     for dataset_dir in sys.argv[1:]:
#     	print(dataset_dir) #print out the filename we are currently processing
#     	day_list = ['otdoor']
#     	for day in day_list:
#             drive_set = os.listdir(dataset_dir + day + '/')
#             for dr in drive_set:
#                 drive_dir = os.path.join(dataset_dir, day, dr)
#                 if os.path.isdir(drive_dir):
#     				ec = read_eventfile(os.path.join(drive_dir, 'events.txt'))
#     				ec.generate_unsync_avg_frame(0.05, drive_dir) #75ms 


# arg[1] - Folder name
# arg[2] - time in seconds(0.01 = 10ms)

def main():
    root = Tk()

    if os.path.isdir(sys.argv[1]):
    	ec = read_eventfile(os.path.join(sys.argv[1], 'events.txt'))
    	ec.generate_evtime(float(sys.argv[2]), sys.argv[1]) #10ms 

    # app = GUI(master=root, event_cloud=ec)
    # root.mainloop()
    # root.destroy()


if __name__ == "__main__":
    main()
