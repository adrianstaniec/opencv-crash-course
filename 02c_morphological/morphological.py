import imutils
import cv2
import numpy as np

more_viz = False

image = cv2.imread('blocks.png')
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)


# thresholding  (binary inverse threshold)
threshed  = cv2.threshold(gray, thresh=250, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Threshed", threshed)
cv2.waitKey(0)

# erosion
mask = threshed.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# dilation
mask = threshed.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

# masking
mask = threshed.copy()
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Masked', masked)
cv2.waitKey(0)

# same thing can be done with just two arguments for bitwise_and,
mask = threshed.copy()
# but we need to first upscale the mask from 1 to 3 channels
mask = mask.reshape(mask.shape[0], -1, 1)  # unsqueeze, to create additonal dimension
mask = np.repeat(mask, 3, axis=2)          # duplicate channels

# masking
masked = cv2.bitwise_and(image, mask)
cv2.imshow('Masked2', masked)
cv2.waitKey(0)
