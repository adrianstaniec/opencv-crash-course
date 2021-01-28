method:

- grayscale -> canny -> findContors -> sort contours
- contours non-max suppresion on area
- find min enclosing rectangles for each contour
- 'calibrate' the scaling ratio on first contour (two euro coin)
- show lines connecting side middlepoints with scaled dimention in milimeters


based on:

Adrian Rosebrock, OpenCV Face Recognition, PyImageSearch
https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
accessed on 28 January 2021
