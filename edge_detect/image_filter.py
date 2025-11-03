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
import sys

# import matplotlib.pyplot as plt

cwd = Path(__file__).resolve().parent
sys.path.append(str(cwd.parent))
from set_logger import set_logger
logger = set_logger(__name__)

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
    hours_from = "hours_from"
    hours_to = "hours_to"
    salary = "salary"
    other = "other"

def taimee(given_image: ndarray | Path | str, thresh_type: int=cv2.THRESH_OTSU, thresh_value: float=150.0, single:bool=False, cvt_color: int=cv2.COLOR_BGR2GRAY, image_dict: dict[ImageDictKey, np.ndarray] | None= None, pre_thresh_valule:float=235.0, binarize:bool=True) -> tuple[float | Sequence[int], np.ndarray]:
    org_image = image_fullpath = None
    match(given_image):
        case ndarray():
            org_image = given_image
        case Path():
            image_fullpath = str(given_image.resolve())
        case str():
            image_fullpath = str(Path(given_image).resolve())
    if image_fullpath is not None:
        org_image = cv2.imread(image_fullpath)
        if org_image is None:
            raise ValueError("Error: Could not load image: %s" % image_fullpath)
    # assert isinstance(image, np.ndarray) #MatLike)
    height, width = org_image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Error: Invalid image shape: %s" % org_image.shape)
    mono_image = cv2.cvtColor(org_image, cvt_color)
    if binarize:
        auto_thresh, pre_image = cv2.threshold(mono_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        auto_thresh = 0
        pre_image = mono_image
    b_image = cv2.threshold(mono_image, thresh=pre_thresh_valule, maxval=255, type=cv2.THRESH_BINARY)[1] # binary, high contrast
    '''fig, ax = plt.subplots(1, 6)
    for r in range(6):
        ax[r].invert_yaxis()
        ax[r].xaxis.tick_top()
    ax[0].imshow(org_image)
    ax[1].imshow(mono_image)
    ax[2].imshow(pre_image)
    ax[3].imshow(b_image)'''
    h_line_ypos_list: list[int] = []
    for ypos in range(height):
        h_line = b_image[ypos, :]
        if (h_line == 0).all(): # len(np.unique(h_line)) == 1 and bool((h_line < 255).any()):
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
    b_image = b_image[last_ypos + 2:, :] # remove pre-heading area and its closing border
    image = pre_image[last_ypos + 2:, :] # remove pre-heading area and its closing border

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

    # mask image of a left-topcv2.threshold(image, thresh=thresh_value, maxval=255, type=thresh_type) circle as blank
    ## find the top block
    cut_height = h_line_ypos_array[0] # h_cur + 1] - last_ypos
    assert cut_height > 0

    h_image = b_image[:cut_height - 1, :]
    ### scan left-top area for a (non-white) shape
    x = -1
    non_unique = False
    for x in (range(width)):
        v_line = h_image[:, x]
        if len(np.unique(v_line)) > 1:
            non_unique = True
            break
    if x == 0:
        raise ValueError("h_image left starts with non-white area!")
    if not non_unique:
        raise ValueError("No shape found in the heading left")
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
    ## erase unwanted h_lines
    for ypos in erase_ypos_list:
        b_image[ypos, :] = 255
    for ypos in ypos_list:
        b_image[ypos, :] = 255
    # Mask the left-top circle as a white rectangle onto image
    cv2.rectangle(image, (0, 0), (cut_x, cut_height), (255, 255, 255), -1)
    if single:
        return auto_thresh, image
    ## get area of hours_from / hours_to
    xpos = -1
    for xpos in reversed(range(width // 2)):
        v_line = b_image[ypos_list[0]:ypos_list[1]-1, xpos]
        if np.count_nonzero(v_line==0) == 0: #len(np.unique(v_line)) == 1 and bool((v_line == 255).all()):
            break
    if xpos == -1:
        raise ValueError("No blank area found at the left side of the hours area center!")
    xpos2 = -1
    for xpos2 in range(width // 2, width):
        v_line = b_image[ypos_list[0]:ypos_list[1]-1, xpos2]
        if np.count_nonzero(v_line==0) == 0: # len(np.unique(v_line)) == 1 and bool((v_line == 255).all()):
            break
    '''b_image[:, xpos2] = 0
    ax[5].imshow(b_image)
    plt.show()
    '''
    if xpos2 == -1 or xpos2 == width or xpos2 <= xpos:
        raise ValueError("No blank area found at the right side of the hours area center!")
    # add the heading area to the dict
    if image_dict is not None:
        image_dict[ImageDictKey.heading] = h_image[:, cut_x + 1:]
        image_dict[ImageDictKey.hours] = image[ypos_list[0]:ypos_list[1], :]
        image_dict[ImageDictKey.hours_from] = image[ypos_list[0]:ypos_list[1], :xpos]
        image_dict[ImageDictKey.hours_to] = image[ypos_list[0]:ypos_list[1], xpos2:]
        image_dict[ImageDictKey.rest_hours] = image[ypos_list[1]:ypos_list[-1], :]
        image_dict[ImageDictKey.other] = image[ypos_list[-1]:, :]

    return ypos_list, image
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