import cv2
import imutils
import numpy as np
from scipy.spatial import distance


img = cv2.imread("things.jpg")
if img is None:
    exit("fail")


cv2.imshow("Things", img)
cv2.waitKey()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_blured = cv2.GaussianBlur(img_gray, (7, 7), 0)
edged = cv2.Canny(img_gray, 60, 200, 3)
cv2.imshow("Edged", edged)
cv2.waitKey()

dilated = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=3)
cv2.imshow("Dilated", dilated)
cv2.waitKey()

eroded = cv2.erode(dilated, np.ones((5, 5), np.uint8), iterations=2)
cv2.imshow("Eroded", eroded)
cv2.waitKey()

contours = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

contoured = img.copy()
contours = list(filter(lambda c: cv2.contourArea(c) > 400, contours))
for contour in contours:
    cv2.drawContours(contoured, [contour], -1, (255, 0, 255), 2)
cv2.imshow("Contoured", contoured)
cv2.waitKey()


def sort_contours(cnts, method="left-to-right"):
    reverse = True if method == "right-to-left" or method == "bottom-to-top" else False
    i = 1 if method == "top-to-bottom" or method == "bottom-to-top" else 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    return zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )


conts, _ = sort_contours(contours)
pixel_per_meter = None


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    tl, bl = leftMost[np.argsort(leftMost[:, 1]), :]

    D = distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    br, tr = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def midpoint(a, b):
    a = np.array(a)
    b = np.array(b)
    return tuple(((a + b) / 2).astype(int))


class Scaler:
    def __init__(self, two_euro_coin_diameter_px):
        two_euro_coin_diameter_mm = 25.75
        self.factor = two_euro_coin_diameter_mm / two_euro_coin_diameter_px

    def __call__(self, size_px):
        size_mm = size_px * self.factor
        return size_mm


scaler = None

for cont in conts:
    if cv2.contourArea(cont) < 100:
        continue

    box = cv2.minAreaRect(cont)
    box = cv2.boxPoints(box)
    box = np.array(box, int)

    box = order_points(box).astype(int)
    cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
    tl, tr, br, bl = box

    top = midpoint(tl, tr)
    bot = midpoint(bl, br)
    left = midpoint(tl, bl)
    right = midpoint(tr, br)

    cv2.line(img, top, bot, (255, 255, 255), 2)
    cv2.line(img, left, right, (255, 255, 255), 2)
    d_height = distance.euclidean(top, bot)
    d_width = distance.euclidean(left, right)

    if scaler is None:
        scaler = Scaler((d_height + d_width) / 2)

    cv2.putText(
        img,
        f"{scaler(d_height):.2f}mm",
        top,
        cv2.FONT_HERSHEY_COMPLEX,
        0.4,
        (255, 255, 255),
    )
    cv2.putText(
        img,
        f"{scaler(d_width):.2f}mm",
        right,
        cv2.FONT_HERSHEY_COMPLEX,
        0.4,
        (255, 255, 255),
    )
    cv2.imshow("box", img)
    cv2.waitKey()

cv2.imwrite("things_measured.jpg", img)