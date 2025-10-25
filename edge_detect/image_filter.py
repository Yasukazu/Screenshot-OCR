from cv2 import UMat
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def taimee(image: UMat | Path | str) -> UMat:
    image_fullpath = None
    match(image):
        case UMat():
            pass
        case Path():
            image_fullpath = image.resolve()
        case str():
            image_fullpath = Path(image).resolve()
    if image_fullpath is not None:
        image = cv2.imread(image_fullpath)
    assert isinstance(image, np.ndarray)
    if image is None:
        raise ValueError("Error: Could not load image: %s" % image_fullpath)
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_lines = []
    for y_coord in range(height):
        h_line = image[y_coord, :]
        if len(np.unique(h_line)) == 1 and bool((h_line < 255).any()):
            h_lines.append(y_coord)
    if len(h_lines) < 2:
        raise ValueError("Not enough heading line found!")
    # cut heading area
    last = h_lines[0]
    for ln, y_coord in enumerate(h_lines[1:]):
        if y_coord > last + 2:
            break
        else:
            last = y_coord
    image = image[last + 1:, :]

    # mask image of a left-top circle as blank
    ## find the top block
    cut_height = h_lines[ln + 1] - last
    assert cut_height > 0
    ### scan left-top area for a (non-white) shape
    for x, x_cd in enumerate(range(width)):
        v_line = image[:(cut_height - 1), x_cd]
        if len(np.unique(v_line)) > 1:
            break
    for x_cd in range(width - x):
        v_line = image[:(cut_height - 1), x_cd + x]
        if len(np.unique(v_line)) == 1 and (v_line == 255).all():
            break
    cut_x = x_cd + x
    # draw a white rectangle
    cv2.rectangle(image, (0, 0), (cut_x, cut_height), (255, 255, 255), -1)
    ret, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    return binary
