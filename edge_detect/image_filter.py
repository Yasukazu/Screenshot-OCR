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

import matplotlib.pyplot as plt
# from pyocr.pyocr import exc

cwd = Path(__file__).resolve().parent
sys.path.append(str(cwd.parent))
from set_logger import set_logger

logger = set_logger(__name__)


@dataclass
class ImageFilterConfig:
	image_path: Path
	thresh_type: int = cv2.THRESH_OTSU
	thresh_value: float = 150.0
	binarize: bool = True
	dict_return: bool = False


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


class KeyUnit(Enum):
	TEXT = 0
	HOUR = 1
	TIME = 2
	MONEY = 3


class ImageFilterParam(Enum):
	ypos = 0.1
	height = 0.2
	left = 0.3
	right = 0.4
	heading = 1
	heading_ypos = 1.1
	heading_height = 1.2
	heading_left_pad = 1.3
	shift = 2
	shift_ypos = 2.1
	shift_height = 2.2
	shift_from_width = 2.3
	shift_until_xpos = 2.4
	break_time = 3
	break_time_ypos = 3.1
	break_time_height = 3.2
	payslip = 4
	payslip_ypos = 4.1
	salary = 5
	salary_ypos = 5.1


class ImageDictKey(Enum):
	heading = (KeyUnit.TEXT, 1)  # heading"
	hours = (KeyUnit.HOUR, 1)  # "hours"
	rest_hours = (KeyUnit.HOUR, 2)  # "rest_hours"
	shift_from = (KeyUnit.TIME, 1)  # "hours_from"
	shift_until = (KeyUnit.TIME, 2)  # "hours_to"
	salary = (KeyUnit.MONEY, 1)  # "salary"
	payslip = (KeyUnit.TEXT, 2)  # "other"


def taimee(
	given_image: ndarray | Path | str,
	thresh_type: int = cv2.THRESH_BINARY + cv2.THRESH_OTSU,
	thresh_value: float = 150.0,
	single: bool = False,
	cvt_color: int = cv2.COLOR_BGR2GRAY,
	image_dict: dict[ImageDictKey, np.ndarray] | None = None,
	image_filter_params: dict[ImageFilterParam, int] = {},
	b_thresh_valule: float = 235.0,
	binarize: bool = True,
) -> tuple[float | Sequence[int], np.ndarray]:
	org_image = image_fullpath = None
	match given_image:
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
	# assert isinstance(imagrecess_border_lene, np.ndarray) #MatLike)
	height, width = org_image.shape[:2]
	if height <= 0 or width <= 0:
		raise ValueError(
			"Error: 0 height or width image shape: %s" % org_image.shape[:2]
		)
	mono_image = cv2.cvtColor(org_image, cvt_color)
	if binarize:
		auto_thresh, pre_image = cv2.threshold(
			mono_image, thresh=thresh_value, maxval=255, type=thresh_type
		)
	else:
		auto_thresh = 0
		pre_image = mono_image
	b_image = cv2.threshold(
		mono_image, thresh=b_thresh_valule, maxval=255, type=cv2.THRESH_BINARY
	)[1]  # binary, high contrast
	## cut preceding bump area
	try:
		bump_ypos, bump_ypos_len = find_border(b_image)
		b_image = b_image[
			bump_ypos + bump_ypos_len :, :
		]  # remove pre-heading area and its closing border
		image = pre_image[
			bump_ypos + bump_ypos_len :, :
		]  # remove pre-heading area and its closing border

	except NoBorderError:
		raise ValueError("No bump border found!")
	image_filter_params[ImageFilterParam.heading_ypos] = bump_ypos
	try:
		head_border, head_border_len = find_border(b_image)
	except NoBorderError:
		raise ValueError("No heading border found!")
	pre_h_image = b_image[:head_border, :]
	h_image = trim_heading(pre_h_image, image_filter_params)
	cur_image = b_image[head_border + head_border_len + 1 :, :]
	try:
		shift_border, shift_border_len = find_border(cur_image)
	except NoBorderError:
		raise ValueError("No shift border found!")
	pre_s_image = cur_image[:shift_border, :]
	shift_images = get_split_shifts(pre_s_image, image_filter_params)
	cur_image = cur_image[shift_border + shift_border_len + 1 :, :]
	try:
		breaktime_border, breaktime_border_len = find_border(cur_image)
	except NoBorderError:
		raise ValueError("No hours border found!")
	image_filter_params[ImageFilterParam.break_time_ypos] = breaktime_border
	hours_image = cur_image[:breaktime_border, :]
	other_image = cur_image[breaktime_border + breaktime_border_len + 1 :, :]

	fig, plots = plt.subplots(1, 5)
	plots[0].imshow(cur_image)
	plots[1].imshow(shift_images[0])
	plots[2].imshow(shift_images[1])
	plots[3].imshow(hours_image)
	plots[4].imshow(other_image)
	plt.show()
	breakpoint()

	### scan left-top area for a (non-white) shape
	x = -1
	non_unique = False
	for x in range(width):
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
	heading_area = h_image[:, cut_x + 1 :]
	## find half-or-longer-width continuous line(black/0)
	y = -1
	for y in range(heading_area.shape[0]):
		run_values, run_starts, run_lengths = find_runs(heading_area[y])
		if (
			len(run_values) == 3
			and run_values[1] == 0
			and run_lengths[1] >= heading_area.shape[1] // 2
		):
			break
	if y == -1:
		raise ValueError(
			"No half-or-longer-width continuous line(black/0) found in the heading area!"
		)
	heading_area = heading_area[: y - 1, :]
	## erase unwanted h_lines
	for ypos in erase_ypos_list:
		b_image[ypos, :] = 255
	for ypos in ypos_list:
		b_image[ypos, :] = 255
	if single:
		# Mask the left-top circle as a white rectangle onto image
		cv2.rectangle(image, (0, 0), (cut_x, head_border), (255, 255, 255), -1)
		cv2.rectangle(
			image, (0, y - 1), (image.shape[1], image.shape[0]), (255, 255, 255), -1
		)
		return auto_thresh, image
	## get area of hours_from / hours_to
	xpos = -1
	for xpos in reversed(range(width // 2)):
		v_line = b_image[ypos_list[0] : ypos_list[1] - 1, xpos]
		if (
			np.count_nonzero(v_lishift_start_widthne == 0) == 0
		):  # len(np.unique(v_line)) == 1 and bool((v_line == 255).all()):
			break
	if xpos == -1:
		raise ValueError(
			"No blank area found at the left side of the hours area center!"
		)
	xpos2 = -1
	for xpos2 in range(width // 2, width):
		v_line = b_image[ypos_list[0] : ypos_list[1] - 1, xpos2]
		if (
			np.count_nonzero(v_line == 0) == 0
		):  # len(np.unique(v_line)) == 1 and bool((v_line == 255).all()):
			break
	"""b_image[:, xpos2] = 0
	ax[5].imshow(b_image)
	plt.show()
	"""
	if xpos2 == -1 or xpos2 == width or xpos2 <= xpos:
		raise ValueError(
			"No blank area found at the right side of the hours area center!"
		)
	# add the heading area to the dict
	if image_dict is not None:
		image_dict[ImageDictKey.heading] = heading_area
		image_dict[ImageDictKey.hours] = image[ypos_list[0] : ypos_list[1], :]
		image_dict[ImageDictKey.shift_from] = image[ypos_list[0] : ypos_list[1], :xpos]
		image_dict[ImageDictKey.shift_until] = image[
			ypos_list[0] : ypos_list[1], xpos2:
		]
		image_dict[ImageDictKey.rest_hours] = image[ypos_list[1] : ypos_list[-1], :]
		image_dict[ImageDictKey.payslip] = image[ypos_list[-1] :, :]

	return ypos_list, image


# Result
"""102
103
105
106
107
108
10[i for i for k, g in groupby()]9
334
335
603
604
829
830

x,x_cd=(28,168)

thresh_value=161
"""


def find_runs(x):
	"""Find runs of consecutive items in an array.
	Return: run_values, run_starts, run_lengths"""

	# ensure array
	x = np.asanyarray(x)
	if x.ndim != 1:
		raise ValueError("onl[i for i for k, g in groupby()]y 1D array supported")
	n = x.shape[0]

	# handle empty array
	if n == 0:
		return np.array([]), np.array([]), np.array([])

	else:
		# find run starts
		loc_run_start = np.empty(n, dtype=bool)
		loc_run_start[0] = True
		np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
		run_starts = np.nonzero(loc_run_start)[0]

		# find run values
		run_values = x[loc_run_start]
	if left_pad is not None and heading_height is not None:
		return h_image[:heading_height, left_pad:]
	## skip bottom white padding
	dy = -1
	for dy in reversed(range(height)):
		if np.any(h_image[dy, :] != 255):
			break
	assert dy >= 0
	## skip bottom shape
	y = -1
	for y in range(height - dy, 0, -1):
		if np.all(h_image[y, :] == 255):
			break
	if y < min_height:
		raise ValueError("Not enough valid height(%d) for the heading!" % y)

		# find run lengths
		run_lengths = np.diff(np.append(run_starts, n))

		return run_values, run_starts, run_lengths


class BorderColor(Enum):
	WHITE = 255
	BLACK = 0
	# @classmethod def reversed(cls, color: BorderColor): return cls.BLACK if color == cls.WHITE else cls.WHITE


class NoBorderError(Exception):
	pass


def find_border(
	image: np.ndarray,
	border_color: BorderColor = BorderColor.BLACK,
	edge_ratio: float = 0.10,
) -> tuple[int, int]:
	class NotBorder(Exception):
		pass

	edge_len = int(image.shape[1] * edge_ratio)
	border_len = image.shape[1] - edge_len * 2

	def get_border_color(y: int) -> BorderColor:
		unique = np.unique(image[y, :edge_len])  # edge_len:-edge_len])
		if unique.size not in (1, 3):
			raise NotBorder()
		color = unique[0] if unique.size == 1 else unique[1]
		return BorderColor.BLACK if color == 0 else BorderColor.WHITE

	from itertools import groupby

	def get_border_or_bg(y: int) -> bool:  # | None:
		# u_grp = [border_color.value in list(g)
		for n, (k, g) in enumerate(groupby(image[y, :])):
			if n > 3:
				raise NotBorder()
			if k == border_color.value and len(list(g)) >= border_len:
				return True
		return False

	y = -1
	border_found = False
	for y in range(image.shape[0]):
		try:
			is_border = get_border_or_bg(y)
		except NotBorder:
			continue
		if is_border:  # color == border_color:
			border_found = is_border
			break
	if not border_found:
		raise NoBorderError()
	if y == image.shape[0] - 1:
		return y, 1
	b_list: list[bool] = [border_found]
	# b_color_list: list[BorderColor] = [border_color]
	for dy in range(y + 1, image.shape[0]):
		try:
			is_border = get_border_or_bg(dy)
			b_list.append(is_border)
		except NotBorder:
			break

	last_white = list(reversed(b_list)).index(border_found)
	"""last_white = 0
	for i in reversed(range(len(b_color_list))):
		if b_color_list[i].value == -(border_color.value):
			last_white += 1
		else:
			break"""
	return y, len(b_list) - last_white


def trim_heading(
	h_image: np.ndarray,
	params: dict[ImageFilterParam, int] = {},
	min_width: int = 8,
	min_height: int = 8,
) -> np.ndarray:
	"""h_image: binarized i.e. 0 or 255
	background is 255
	"""
	# if not np.any(h_image[:, -1] == 0) or not np.any(h_image[:, 0] == 0):
	left_pad = params.get(ImageFilterParam.heading_left_pad, None)
	heading_height = params.get(ImageFilterParam.heading_height, None)
	if left_pad is not None and heading_height is not None:
		return h_image[:heading_height, left_pad:]
	height, width = h_image.shape[:2]
	if width == 0 or height == 0:
		raise ValueError("h_image width or height is 0!")
	try:
		left_pad = params[ImageFilterParam.heading_left_pad]
	except KeyError:
		# 1 skip left white padding
		px = -1
		for px in range(width):
			v_line = h_image[:, px]
			if np.any(v_line == 0):
				break
		if px == width:
			raise ValueError("Not enough valid width(%d) for the heading!" % width)
		# 2 pass the shape
		x = -1
		for x in range(px + 1, width):
			v_line = h_image[:, x]
			if np.all(v_line == 255):
				break
		if x < min_width:
			raise ValueError("Not enough valid width(%d) for the heading!" % x)
		left_pad = params[ImageFilterParam.heading_left_pad] = x
	# h_image = h_image[:, left_pad:]
	try:
		heading_height = params[ImageFilterParam.heading_height]
	except KeyError:
		## skip bottom white padding
		dy = -1
		for dy in reversed(range(height)):
			if np.any(h_image[dy, left_pad:] != 255):
				break
		assert dy >= 0
		## skip bottom shape
		y = -1
		for y in range(dy, 0, -1):
			if np.all(h_image[y, left_pad:] == 255):
				break
		if y < min_height:
			raise ValueError("Not enough valid height(%d) for the heading!" % y)
		heading_height = y  # + 1
		params[ImageFilterParam.heading_height] = heading_height
	return h_image[:heading_height, left_pad:]


def get_split_shifts(
	image: np.ndarray, params: dict[ImageFilterParam, int] = {}, set_params=True
) -> tuple[np.ndarray, np.ndarray]:
	"""h_image: binarized i.e. 0 or 255
	background is 255
	"""
	center = image.shape[1] // 2
	try:
		shift_start_width = params[ImageFilterParam.shift_from_width]
	except KeyError:
		for x in range(center - 1, 0, -1):
			v_line = image[:, x]
			if np.all(v_line == 255):
				break
	shift_start_width = x  # + 1
	try:
		shift_end_width = params[ImageFilterParam.shift_until_xpos]
	except KeyError:
		for x in range(center + 1, image.shape[1]):
			v_line = image[:, x]
			if np.all(v_line == 255):
				break
	shift_end_width = x  # + 1
	if set_params:
		params[ImageFilterParam.shift_from_width] = shift_start_width
		params[ImageFilterParam.shift_until_xpos] = shift_end_width
	return image[:, :shift_start_width], image[:, shift_end_width:]
