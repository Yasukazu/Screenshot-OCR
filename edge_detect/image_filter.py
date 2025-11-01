from cv2 import UMat
import cv2
# from cvc2.typing import MatLike
# import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from pathlib import Path
from typing import Sequence
from dataclasses import dataclass
from enum import Enum

@dataclass
class ImageFilterConfig:
    image_path: Path
    thresh_type: int=cv2.THRESH_OTSU
    thresh_value: float=150.0
    binarize: bool=True
    dict_return: bool=False

@dataclass
class SalaryStatementImages:
    heading: UMat
    time_from: UMat
    time_to: UMat
    salary: UMat
    other: UMat

@dataclass
class ImageFilterResult:
    h_lines: Sequence[int]
    filtered_image: UMat
    image_dict: SalaryStatementImages | None = None
    thresh_value: float = 0.0

class ImageDictKey(Enum):
    heading = "heading"
    hours = "hours"
    rest_hours = "rest_hours"
    time_from = "time_from"
    time_to = "time_to"
    salary = "salary"
    other = "other"

def taimee(given_image: ndarray | Path | str, thresh_type: int=cv2.THRESH_OTSU, thresh_value: float=150.0, binarize=True, cvt_color: int=cv2.COLOR_BGR2GRAY, image_dict: dict[ImageDictKey, np.ndarray] | None= None) -> tuple[float | Sequence[int], np.ndarray]:
    image_fullpath = image = None
    match(given_image):
        case ndarray():
            image = given_image
        case Path():
            image_fullpath = str(given_image.resolve())
        case str():
            image_fullpath = str(Path(given_image).resolve())
    if image_fullpath is not None:
        image = cv2.imread(image_fullpath)
        if image is None:
            raise ValueError("Error: Could not load image: %s" % image_fullpath)
    # assert isinstance(image, np.ndarray) #MatLike)
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cvt_color)
    h_line_ypos_list: list[int] = []
    for ypos in range(height):
        h_line = image[ypos, :]
        if len(np.unique(h_line)) == 1 and bool((h_line < 255).any()):
            h_line_ypos_list.append(ypos)
    if len(h_line_ypos_list) < 2:
        raise ValueError("Not enough heading line found!")
    # cut heading area
    last_ypos = h_line_ypos_list[0]
    h_cur = -1
    for h_cur, ypos in enumerate(h_line_ypos_list[1:]):
        if ypos > last_ypos + 2:
            break
        else:
            last_ypos = ypos
    assert h_cur >= 0
    # heading_ypos = last_ypos + 1
    image = image[last_ypos + 1:, :] # remove pre-heading area and its closing border
    h_line_ypos_array = np.array(h_line_ypos_list[h_cur + 1:]) - (last_ypos + 1)
    last_ypos = h_line_ypos_array[0]
    ypos_list: list[int] =[last_ypos]
    ypos = h_cur = -1
    erase_ypos_list = []
    for h_cur, ypos in enumerate(h_line_ypos_array[1:]):
        if ypos > ypos_list[-1] + 1:
            ypos_list.append(ypos)
        else:
            last_ypos = ypos
            erase_ypos_list.append(ypos)
    if len(ypos_list) == 1: # h_cur < 0 or ypos < 0:
        raise ValueError("Failed to find the next ypos!")

    
    # mask image of a left-top circle as blank
    ## find the top block
    cut_height = h_line_ypos_array[0] # h_cur + 1] - last_ypos
    assert cut_height > 0
    h_image = image[:cut_height - 1, :]
    ### scan left-top area for a (non-white) shape
    x = -1
    unique_found = False
    for x in (range(width)):
        v_line = h_image[:, x]
        if len(np.unique(v_line)) > 1:
            unique_found = True
            break
    if x == -1:
        raise ValueError("width is 0!")
    if not unique_found:
        raise ValueError("No shape found in the heading liath")
### scan left-top area for (white) area
    x_cd = -1
    blank_area_found = False
    for x_cd in range(width - x):
        v_line = h_image[:, x_cd + x]
        if len(np.unique(v_line)) == 1 and (v_line == 255).all():
            blank_area_found = True
            break
    if x_cd == -1 or not blank_area_found:
        raise ValueError("No blank area found at the right side of the heading shape!")
    cut_x = x_cd + x
    # add the heading area to the dict
    if image_dict is not None:
        image_dict[ImageDictKey.heading] = h_image[:, cut_x + 1:]
        image_dict[ImageDictKey.hours] = image[ypos_list[0]:ypos_list[1], :]
        image_dict[ImageDictKey.rest_hours] = image[ypos_list[1]:ypos_list[-1], :]
        image_dict[ImageDictKey.other] = image[ypos_list[-1]:, :]
    # draw a white rectangle
    cv2.rectangle(image, (0, 0), (cut_x, cut_height), (255, 255, 255), -1)
    if not binarize:
        return ypos_list, image
    else:
        return cv2.threshold(image, thresh=thresh_value, maxval=255, type=thresh_type) 
# Result
'''102
103
105
106
107
108
109
334
335
603
604
829
830

x,x_cd=(28,168)

thresh_value=161
'''