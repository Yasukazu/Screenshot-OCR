# The result of rect_detect.py

This script succeeded to detect rectangles in the parametarized ranges, about complete rectangle shape in the image.
The `Rect` object returned by `boundingRect(contour)` has [w, h, x, y] attributes.

## Options

- `-a`: max aspect ratio
  + default: 10.0
    - set to 1.05 to detect nearly-perfect squares
- `-t`: threshold ratio
  + default: 0.5
    - set to 0.95 to detect grayish shapes

## In-main-script parameters

- `contour_limit`: Number of contours to limit.


## Speed

Within 1 second to find rectangles in the parametarized ranges.

## Reference(s) of `findContours()`

[Detect contours by OpenCV with Python | CodeVace](https://www.codevace.com/py-opencv-findcontours/)
