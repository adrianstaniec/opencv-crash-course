from random import randint
import cv2
import numpy as np
import imutils
# threshold_adaptive is obsolete, import theshold_local
from skimage.filters import threshold_local  
from transform import four_point_transform

args = {}
args['image'] = 'paragon.jpg'
image = cv2.imread(args['image'])

REDUCED_HEIGHT = 500
scale_factor = image.shape[0] / REDUCED_HEIGHT
print(id(image))
image_small = imutils.resize(image, height=REDUCED_HEIGHT)
print(id(image_small))
cv2.imshow('Document', image_small)
cv2.waitKey(0)

greys = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(greys, (5,5), 0)
edges = cv2.Canny(greys,  20, 230)

cv2.imshow('Edges', edges)
cv2.waitKey(0)


contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

print('# found contours:', len(contours))
contours = [c for c in contours if cv2.contourArea(c) > 100]
print('# found contours:', len(contours), 'with area above 100')

for i, c in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
    peri = cv2.arcLength(c, True)
    approx_vertices = cv2.approxPolyDP(c, 0.02 * peri, True)
    print(f'Contour {i} area:', cv2.contourArea(c), 'arcLength:', peri, 'approx:', approx_vertices)
    
    if len(approx_vertices) == 4:
        # this method finds a lot of duplicates,
        # so just be satisfied with the first one
        break

random_color = (randint(0,255), randint(0,255), (randint(0,255)))
cv2.drawContours(image_small, [approx_vertices], -1, random_color, 2)
cv2.imshow("Outline", image_small)
cv2.waitKey(0)

warped_small = four_point_transform(greys, approx_vertices.reshape(4,2))
cv2.imshow("Warped edges", warped_small)
cv2.waitKey(0)

warped = four_point_transform(image, approx_vertices.reshape(4,2) * scale_factor)
cv2.imshow("Warped original", warped)
cv2.waitKey(0)

warped_grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped_grey, 55, offset=20, method='gaussian')
threshed = (warped_grey > T).astype(np.uint8) * 255
cv2.imshow("Warped, greyed and thresholded", threshed)
cv2.waitKey(0)
