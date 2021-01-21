import imutils
import cv2
import numpy as np

more_viz = False

image = cv2.imread('blocks.png')
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
if more_viz:
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)

# edge detection
edges = cv2.Canny(gray, threshold1=30, threshold2=150)
if more_viz:
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)

# thresholding  (binary inverse threshold)
threshed  = cv2.threshold(gray, thresh=250, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
if more_viz:
    cv2.imshow("Threshed", threshed)
    cv2.waitKey(0)

# contours
COLOR = (0, 100, 200)
contours = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

output = image.copy()
for c in contours:
    cv2.drawContours(output, [c], -1, COLOR, 3)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)
cv2.putText(output, f"found {len(contours)} objects!" , (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)
