import cv2;
import numpy as np;

# Read image
img = cv2.imread("label_00000054.png", cv2.IMREAD_GRAYSCALE);

# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.
kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(img,kernel,iterations = 2)

# Display images.

cv2.imshow("Foreground", dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()