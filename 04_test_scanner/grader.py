from random import randint

import cv2
import numpy as np
import imutils
import imutils.contours
from imutils.perspective import four_point_transform

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

DEBUG = True

CORRECT_ANSWERS = {1: "A", 2: "C", 3: "D", 4: "E", 5: "B"}


def extract_edges_from_grey_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.Canny(img, 50, 200)


def extract_contours_from_edge_img(img):
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(contours)


def suppress_all_but_n_max_area_contours_and_sort_them(contours, n):
    return [
        c
        for i, c in enumerate(sorted(contours, key=cv2.contourArea, reverse=True))
        if i < n
    ]


def extract_largest_quadrangle_from_contours(contours):
    big_sorted_contours = suppress_all_but_n_max_area_contours_and_sort_them(
        contours, 5
    )
    for c in big_sorted_contours:
        peri = cv2.arcLength(c, True)
        approx_vertices = cv2.approxPolyDP(c, 0.02 * peri, True)

        if DEBUG:
            print(f"Contour area:", cv2.contourArea(c))
            print("arcLength:", peri)
            print("approx:", approx_vertices)

        if len(approx_vertices) == 4:
            # this method finds a lot of duplicates,
            # just be satisfied with the first one than has 4 corners
            break
    return approx_vertices


def random_color():
    return (randint(0, 255), randint(0, 255), (randint(0, 255)))


def straighten_and_crop(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_img = extract_edges_from_grey_image(grey)
    contours = extract_contours_from_edge_img(edge_img)
    vertices = extract_largest_quadrangle_from_contours(contours)

    if DEBUG:
        cv2.drawContours(img, [vertices], -1, (255, 0, 0), 2)
        cv2.imshow("Found document", img)
        cv2.waitKey(0)

    img_warped = four_point_transform(img, vertices.reshape(4, 2))
    grey_warped = four_point_transform(grey, vertices.reshape(4, 2))

    if DEBUG:
        cv2.imshow("Warped document", img_warped)
        cv2.waitKey(0)

    return img_warped, grey_warped


def extract_anwers_from_img(img):
    doc, doc_grey = straighten_and_crop(img)

    # Otsu Method (1979) for automatic thresholding
    # nice explanation: https://www.youtube.com/watch?v=0DFdT0OWLeo
    doc_bw = cv2.threshold(doc_grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # NOTE: for some reason, the contour finding fails when the image
    # the image is first warped then converted to gray scale
    # so first convert to greayscale then warp
    contours = cv2.findContours(
        doc_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)

    if DEBUG:
        cv2.drawContours(doc, contours, -1, RED, 2)
        cv2.imshow("all contours", doc)
        cv2.waitKey(0)

    def is_bubble_contour(contour):
        min_bubble_diameter = 20
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if (
            w >= min_bubble_diameter
            and h >= min_bubble_diameter
            and aspect_ratio >= 0.9
            and aspect_ratio <= 1.1
        ):
            return True
        else:
            return False

    bubble_contours = [c for c in contours if is_bubble_contour(c)]
    bubble_contours_vsorted = imutils.contours.sort_contours(
        bubble_contours, method="top-to-bottom"
    )[0]

    if DEBUG:
        cv2.drawContours(doc, bubble_contours_vsorted, -1, GREEN, 2)
        cv2.imshow("bubble contours", doc)
        cv2.waitKey(0)

    def divide_rows_and_sort_each(contours_vsorted):
        n_questions = len(contours_vsorted) // 5
        contours_vsorted_vgrouped_hsorted = [
            imutils.contours.sort_contours(contours_vsorted[i * 5 : (i + 1) * 5])[0]
            for i in range(n_questions)
        ]

        if DEBUG and False:
            for cnts in contours_vsorted_vgrouped_hsorted:
                row_color = random_color()
                for c in cnts:
                    cv2.drawContours(doc, [c], -1, row_color, 2)
                    cv2.imshow("row by row bubbles", doc)
                    cv2.waitKey(0)

        return contours_vsorted_vgrouped_hsorted

    bubble_contours_grouped_and_sorted = divide_rows_and_sort_each(
        bubble_contours_vsorted
    )

    def extract_answer_of_one_question(img_bw, row):
        pixels_colored = []
        for bubble in row:
            mask = np.zeros(img_bw.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble], -1, 255, cv2.FILLED)
            mask = cv2.bitwise_and(img_bw, img_bw, mask=mask)
            total = cv2.countNonZero(mask)
            pixels_colored.append(total)
        return np.argmax(pixels_colored)

    extracted_answers = [
        extract_answer_of_one_question(doc_bw, row)
        for row in bubble_contours_grouped_and_sorted
    ]

    for row, answer in zip(bubble_contours_grouped_and_sorted, extracted_answers):
        cv2.drawContours(doc, [row[answer]], -1, (200, 000, 200), 2)
    cv2.imshow("bubbled", doc)
    cv2.waitKey(0)


    return extracted_answers


def grade(answers, correct_answers):
    score = 0
    for i, given in enumerate(answers, 1):
        if given == correct_answers[i]:
            score += 1
    return score


def main():
    image = cv2.imread("tests.jpg")
    h, w, c = image.shape
    tests = [
        image[: h // 2, : w // 2],
        image[: h // 2, w // 2 :],
        image[h // 2 :, 0 : w // 2],
        image[h // 2 :, w // 2 :],
    ]

    print("Correct answers are:")
    for k, v in CORRECT_ANSWERS.items():
        print(f"- Question #{k}: {v}")

    for i, test in enumerate(tests, 1):
        answers = extract_anwers_from_img(test)
        answers = [["A", "B", "C", "D", "E"][i] for i in answers]
        score = grade(answers, CORRECT_ANSWERS)
        print(f"\nStudent #{i}:")
        print(f"  answers: {answers}")
        print(f"  score:   {score}/5")
        global DEBUG
        DEBUG = False


if __name__ == "__main__":
    main()
