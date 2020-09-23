import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

import cv2
import numpy as np


PROTOTXT = './deploy.prototxt'
MODEL = './res10_300x300_ssd_iter_140000.caffemodel'
IMAGE = './image.jpg'
CONFIDENCE_TH = 0.5
BBOX_COLOR = (0, 0, 255)

logging.info("loading model ...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

image = cv2.imread(IMAGE)
h, w = image.shape[:2]

def preprocess_image(image, new_size):
    small_img = cv2.resize(image, new_size)
    return cv2.dnn.blobFromImage(small_img, 1.0, new_size, (104, 177, 123))


new_size = 400, 400
blob = preprocess_image(image, new_size)

logging.info("computing detections...")
net.setInput(blob)
detections = net.forward()

logging.info("drawing detections...")
for i in range(detections.shape[2]):
    det = detections[0, 0, i, :]
    confidence = det[2]
    box = det[3:7]
    if confidence < CONFIDENCE_TH:
        continue
    print("Dettecion", i, "\tConfidence:", confidence)
    box = box * np.array([w, h, w, h])
    x1, y1, x2, y2 = box.astype("int")
    cv2.rectangle(image, (x1, y1), (x2, y2), BBOX_COLOR, 1)
    cv2.putText(
        image,
        f"{confidence*100:.0f}%",
        (x2, y2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        BBOX_COLOR,
        1,
    )

cv2.imshow("Face Detection", image)
cv2.waitKey(0)
