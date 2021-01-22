from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from dataclasses import dataclass


lowerb = (35, 65, 100)
upperb = (55, 255, 255)

centers = deque(maxlen=128)
vs = VideoStream(src=0).start()

time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    frame = cv2.flip(frame, 1)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lowerb, upperb)
    mask0 = mask.copy()
    kernel = np.ones((5, 5), dtype=np.uint8)
    kernel[0, 0] = 0
    kernel[0, 4] = 0
    kernel[4, 0] = 0
    kernel[4, 4] = 0
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    center = None
    if len(cnts) > 0:
        max_cnt = max(cnts, key=cv2.contourArea)
        center, r = cv2.minEnclosingCircle(max_cnt)
        if 50 < r and r < 200:
            center = tuple(map(int, center))
            r = int(r)
            print(center, r)
            cv2.circle(frame, center, r, (100, 100, 250), 2)
        else:
            center = None
    centers.append(center)

    for i in range(1, len(centers)):
        if centers[i - 1] is not None and centers[i] is not None:
            thickness = int(np.ceil(i / 10))
            print(centers[i], thickness)
            cv2.line(frame, centers[i - 1], centers[i], (250, 100, 100), thickness)

    cv2.imshow("Frame", np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()

