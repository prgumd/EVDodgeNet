import cv2
import numpy as np
import sys

# argv is your commandline arguments, argv[0] is your program name, so skip it
# for n in sys.argv[1:]:
#     print(n) #print out the filename we are currently processing
#     input = open(n, "r")
#     output = open(n + ".out", "w")
#     # do some processing
#     input.close()
#     output.close()

for n in range(10000):
	n = n+1
	# link = os.path.join(sys.argv[1],'frame_%08d.png')%n
	# print(link)

	img1 = cv2.imread('/media/analogicalnexus/00EA777C1E864BA9/2018/simulator_dataset/3/images/frame_%08d.png'%n,0)
	img2 = cv2.imread('/media/analogicalnexus/00EA777C1E864BA9/2018/simulator_dataset/3_wo/images/frame_%08d.png'%n,0)

	# image subtraction
	img3 = img1-img2
	ret,thresh1 = cv2.threshold(img3,0,255,cv2.THRESH_BINARY)

	# hole filling
	kernel = np.ones((5,5),np.uint8)
	thresh1 = cv2.dilate(thresh1,kernel,iterations = 2)
	# h, w = thresh1.shape[:2]
	# mask = np.zeros((h+2, w+2), np.uint8)
	# # Floodfill from point (0, 0)
	# cv2.floodFill(thresh1, mask, (0,0), 255);
	cv2.imwrite('/media/analogicalnexus/00EA777C1E864BA9/2018/simulator_dataset/3_mask/mask_%08d.png'%n, thresh1)
	# cv2.imshow('result',thresh1)
	# cv2.waitKey()
	# cv2.destroyAllWindows()