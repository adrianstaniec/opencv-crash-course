from random import randint

import cv2
import numpy as np
import imutils
from imutils import contours
from imutils.perspective import four_point_transform

image = cv2.imread("tests.jpg")
h, w, c = image.shape
tests = [
    image[: h // 2, : w // 2],
    image[: h // 2, w // 2 :],
    image[h // 2 :, 0 : w // 2],
    image[h // 2 :, w // 2 :],
]

for i, test in enumerate(tests):
    cv2.imshow(f'Test {i}', test)
    cv2.waitKey(0)

CORRECT_ANSWERS = {1: 'D', 2: 'E', 3: 'A', 4: 'D', 5: 'A'}

print("Correct answers are:")
for k,v in CORRECT_ANSWERS.items():
    print(f"- Question #{k}: {v}")


def extract_edges_from_color_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.Canny(img, 50, 200)

def extract_contours_from_edge_img(img):
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(contours)

def suppress_all_but_n_max_area_contours_and_sort_them(contours, n):
    return [c for i, c in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)) if i < n]
    
def extract_largest_quadrangle_from_contours(contours):
    big_sorted_contours = suppress_all_but_n_max_area_contours_and_sort_them(contours, 5)
    for c in big_sorted_contours:
        peri = cv2.arcLength(c, True)
        approx_vertices = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(f'Contour area:', cv2.contourArea(c), 'arcLength:', peri, 'approx:', approx_vertices)
        if len(approx_vertices) == 4:
            # this method finds a lot of duplicates,
            # just be satisfied with the first one than has 4 corners
            break
    return approx_vertices

def random_color():
    return (randint(0,255), randint(0,255), (randint(0,255)))

def straighten_and_crop(img):
    edge_img = extract_edges_from_color_image(img)
    contours = extract_contours_from_edge_img(edge_img)
    vertices = extract_largest_quadrangle_from_contours(contours)

    # visualization only
    cv2.drawContours(img, [vertices], -1, (255, 0, 0), 2)
    cv2.imshow('Found document', img)
    cv2.waitKey(0)

    img_warped = four_point_transform(img, vertices.reshape(4,2))

    # visualization only
    cv2.imshow('Warped document', img_warped)
    cv2.waitKey(0)

    return img_warped


def extract_anwers_from_img(img):
    doc = straighten_and_crop(img)
    doc_grey = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)
    doc_bw = cv2.threshold(doc_grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # visualization only
    cv2.imshow('BW document', doc_bw)
    cv2.waitKey(0)

    return []

def grade(img, correct_answers):
    answers = extract_anwers_from_img(img)
    return 0

for test in tests:
    grade(test, CORRECT_ANSWERS)
