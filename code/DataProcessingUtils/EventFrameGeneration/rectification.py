import cv2
import numpy as np

img = cv2.imread('frame_otdoor.png')
h,w = img.shape[:2]
226.38018519795807, 226.15002947047415, 173.6470807871759, 133.73271487507847
K = np.array([[226.380, 0., 173.647], [0., 226.150, 133.732], [0., 0., 1.]], dtype=np.float32)
D = np.array([-0.033904378348448685, -0.01537260902537579, -0.022284741346941413, 0.0069204143687187645], dtype=np.float32)
# 226.0181418548734, 225.7869434267677, 174.5433576736815, 124.21627572590607
# K = np.array([[226.018, 0., 174.543], [0., 225.786, 124.216], [0., 0., 1.]], dtype=np.float32)
# D = np.array([-0.04846669832871334, 0.010092844338123635, -0.04293073765014637,
    # 0.005194706897326005], dtype=np.float32)

Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K,D,(w,h),None)
print(Knew)
# Knew[(0,1), (0,1)] = .3* Knew[(0,1), (0,1)]

mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K,D,None,Knew,(w,h),5)
img_undistorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
crop_img = img_undistorted[:180, :]
cv2.imshow("cropped", crop_img)
print(img_undistorted.shape)

cv2.imshow('undistorted', img_undistorted)
# cv2.imshow('distorted', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# undistorted = cv2.fisheye.undistortImage(distorted_image, K, D)