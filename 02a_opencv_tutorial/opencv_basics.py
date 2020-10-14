import imutils
import cv2

IMAGE = "./image.jpg"

image = cv2.imread(IMAGE)
cv2.imshow("Image", image)
cv2.waitKey(0)

# %% basic params

# get image size
height, width, depth = image.shape
print(f'width={width}, height={height}, depth={depth}')

# get pixel values
b,g,r = image[150,300]
print(f'blue={b}, green={g}, red={r}')

# %% slicing / cropping

roi = image[150:250]
cv2.imshow("Row Crop", roi)
cv2.waitKey(0)

roi = image[:, 150:300]
cv2.imshow("Column Crop", roi)
cv2.waitKey(0)

roi = image[150:300, 150:300]
cv2.imshow("Rectangle Crop", roi)
cv2.waitKey(0)

roi = image[:,:,2]
cv2.imshow("Channel Crop", roi)
cv2.waitKey(0)

# %% resizing

resized = cv2.resize(image, (200, 200))
cv2.imshow('Resizing', resized)
cv2.waitKey(0)

aspect_ratio = 200.0 / width
new_dims = (200, int(height*aspect_ratio))
resized = cv2.resize(image, new_dims)
cv2.imshow('Fixed Aspect aspect_ratio Resizing', resized)
cv2.waitKey(0)

## same but shorter
resized = imutils.resize(image, width=200)
cv2.imshow('Imutils Resize', resized)
cv2.waitKey(0)

# %% rotating

center = (width//2,height//2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (width, height))
cv2.imshow('Rotated', rotated)
cv2.waitKey(0)

## same but shorter
rotated = imutils.rotate(image, -45)
cv2.imshow('Imutils rotated', rotated)
cv2.waitKey(0)

## same but extended instead of unclipped
rotated = imutils.rotate_bound(image, -45)
cv2.imshow('Imutils bound rotated', rotated)
cv2.waitKey(0)

# %% smooting
blurred = cv2.GaussianBlur(image, (11,11), 0)
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)


# %% drawing on image
output = image.copy()
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
LINE_THICKNESS = 2
cv2.rectangle(output, (200, 50), (400, 150), RED_COLOR, LINE_THICKNESS)
cv2.circle(output, (300, 200), 20, GREEN_COLOR, -1)
cv2.line(output, (60, 200), (400, 400), BLUE_COLOR, 5)
FONT_SCALE = 0.7
cv2.putText(output, "OpenCV!!!", (400, 50), 
	cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, RED_COLOR, LINE_THICKNESS)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)