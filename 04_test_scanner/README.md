method:
- fix the perspective
    - threshold -> canny -> find largest quadrangle contour -> perspective warp to a rectangle

- find bubbles
    - otsu threshold -> find not too small contours that can be inscribedin a square
    - sort top-to-bottom and divide into groups
    - sort each group left-to-right

- check if bubbled
    - draw filled contour mask the negative with the image
    - threshold it
    - count number of pixels
    - pick maximum as an answer

- compare to the answer key


based on:

Adrian Rosebrock, OpenCV Face Recognition, PyImageSearch
https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/
accessed on 20 October 2020
