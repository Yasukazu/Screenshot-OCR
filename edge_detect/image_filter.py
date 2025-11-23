from io import IOBase
from dataclasses import field

from cv2 import UMat
import cv2
from tesseract_ocr import TesseractOCR

# from cvc2.typing import MatLike
# import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from pathlib import Path
from typing import Iterator, Sequence, NamedTuple, override, Sequence
from dataclasses import dataclass
from enum import Enum
import sys
import matplotlib.pyplot as plt
from inspect import isclass
from itertools import groupby
from fancy_dataclass import TOMLDataclass

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
	PIXEL = -1
	TEXT = 0
	HOUR = 1
	TIME = 2
	MONEY = 3


class ImageDictKey(Enum):
	leading = (KeyUnit.PIXEL, 1)
	heading = (KeyUnit.TEXT, 1)  # heading"
	work_time = (KeyUnit.HOUR, 1)  # "hours"
	break_time = (KeyUnit.HOUR, 2)  # "rest_hours"
	shift_start = (KeyUnit.TIME, 1)  # "hours_from"
	shift_end = (KeyUnit.TIME, 2)  # "hours_to"
	salary = (KeyUnit.MONEY, 1)  # "salary"
	payslip = (KeyUnit.TEXT, 2)  # "other"

class InvalidValueError(ValueError):
	"""Raised when a value is invalid."""
	pass

class ItemAreaParam(NamedTuple):
	ypos: int = 0
	height: int = -1
	xpos: int = 0
	width: int = -1

# from Python 3.12
type Int4 = tuple[int, int, int, int]
# type ItemAreaParamType = Int4 | Sequence[Int4]

@dataclass
class ImageFilterItemArea(TOMLDataclass):
	ypos: int = 0
	height: int = -1
	xpos: int = 0
	width: int = -1

	# param: ItemAreaParam # NamedTuple:read only

	def __post_init__(self):
		if self.ypos < 0 or self.xpos < 0:
			raise InvalidValueError("ypos and xpos must be positive")
		if (self.height < -1 or self.height == 0) or (self.width < -1 or self.width == 0):
			raise InvalidValueError("height and width must be larger than 0 except -1")

	@property
	def param(self)-> Int4:
		return (self.ypos, self.height, self.xpos, self.width)


	def as_slice_param(self) -> Sequence[Int4]:
		return ((self.ypos, (self.ypos + self.height) if self.height > 0 else -1, self.xpos, (self.xpos + self.width) if self.width > 0 else -1),)

	def crop_image(self, image: np.ndarray) -> Iterator[np.ndarray]:
		for param in self.as_slice_param():
			yield image[param[0]:param[1], param[2]:param[3]]


@dataclass
class LeadingArea:
	height: int

@dataclass
class HeadingArea(ImageFilterItemArea):
	'''Necessary named parameters: ypos, height, xpos '''
	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(f"{self.__class__.__name__} = {str(list(self.param))}\n")


@dataclass
class ShiftArea(ImageFilterItemArea):
	'''Needs to initialize using named parameters::
	start_width: as xpos
	end_xpos: as width
	while start_xpos is 0 and end_width is -1
	'''

	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(f"{self.__class__.__name__} = {str(list(self.param))}\n")
	@property
	def start_width(self)-> int: # start-from time
		return self.xpos

	@property
	def end_xpos(self)-> int: # end-by time
		return self.width
@dataclass
class ShiftStartArea(ImageFilterItemArea):
	pass

@dataclass
class ShiftEndArea(ImageFilterItemArea):
	pass

	def crop_image(self, image: np.ndarray) -> Iterator[np.ndarray]:
		for param in self.as_slice_param():
			yield image[param[0]:param[1], param[2]:param[3]]

@dataclass
class BreaktimeArea(ImageFilterItemArea):
	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(f"{self.__class__.__name__} = {str(list(self.param))}\n")

@dataclass
class PayslipArea(ImageFilterItemArea):
	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(f"{self.__class__.__name__} = {str(list(self.param))}\n")
@dataclass
class SalaryArea(ImageFilterItemArea):
	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(f"{self.__class__.__name__} = {str(list(self.param))}\n")


@dataclass
class ImageFilterAreas:
	'''tuple's first element is ypos (downward offset from heading top) and second element is height
	'''
	def to_toml(self, fp: IOBase, **kwargs):
		'''requires "name" key in kwargs'''
		if 'name' not in kwargs:
			raise ValueError("requires 'name' key in kwargs")
		if not kwargs['name']:
			raise ValueError("requires 'name' key's value in kwargs")
		fp.write(f"[{self.__class__.__name__}.{kwargs['name']}]\n")
		try:
			as_dict = kwargs['as_dict']
		except KeyError:
			as_dict = False
		for key, area in self.__dict__.items():
			if isclass(area): # area.to_toml(fp)
				fp.write(f"{key} = ")
				fp.write(f"{area.as_dict()}\n") if as_dict else fp.write(f"{list(area.param)}\n")

	area_key_list = [ImageDictKey.leading, ImageDictKey.heading, ImageDictKey.shift_start, ImageDictKey.shift_end, ImageDictKey.break_time, ImageDictKey.payslip, ImageDictKey.salary]

	areas = {
		ImageDictKey.leading: LeadingArea,
		ImageDictKey.heading: HeadingArea,
		ImageDictKey.shift_start: ShiftStartArea,
		ImageDictKey.shift_end: ShiftEndArea,
		ImageDictKey.break_time: BreaktimeArea,
		ImageDictKey.payslip: PayslipArea,
		ImageDictKey.salary: SalaryArea,
	}
	heading: HeadingArea # midashi
	shift: ShiftArea # syuugyou jikan
	break_time: BreaktimeArea # kyuukei jikan
	payslip: PayslipArea # meisai
	salary: SalaryArea # kyuuyo
	y_offset: int = 0



class ImageFilterParam(Enum):
	leading = 0, 0
	ypos = 0,1
	height = 0,2
	xpos = 0,3
	width = 0,4
	heading = 1, 0
	heading_ypos = 1,1
	heading_height = 1,2
	heading_xpos = 1,3
	shift = 2, 0
	shift_ypos = 2,1
	shift_height = 2,2
	shift_start_width = 2,3
	shift_end_xpos = 2,4
	breaktime = 3, 0
	breaktime_ypos = 3,1
	breaktime_height = 3,2
	paystub = 4, 0
	paystub_ypos = 4,1
	salary = 5, 0
	salary_ypos = 5,1


class NonNearbyElems:
	def __init__(self,
		thresh: int = 10,
		elems: list[int] = []
	):
		self.thresh = thresh
		self.elems = elems

	def add(self, i: int):
		if len(self.elems) == 0:
			self.elems.append(i)
		else:
			if i - self.elems[-1] > self.thresh:
				self.elems.append(i)

class BorderColor(Enum):
	WHITE = 255
	BLACK = 0


class NoBorderError(Exception):
	pass
def find_horizontal_borders(
	image: np.ndarray,
	border_color: BorderColor = BorderColor.BLACK,
	edge_ratio: float = 0.10,
) -> Iterator[int]:
	"""Find border lines in the image.
	Return: list of border ypos"""


	edge_len = int(image.shape[1] * edge_ratio)
	border_len = image.shape[1] - edge_len * 2


	def get_border_or_bg(y: int) -> bool | None:
		# Returns True if border is found, False if background is found, else returns None
		arr = image[y, edge_len:-edge_len]
		if np.all(arr == border_color.value):
			return True
		elif np.all(arr == 255):
			return False
		return None
		'''changes = arr[1:] != arr[:-1]
		change_indices = np.where(changes)[0] + 1
		if not(change_indices.size):
			return None
		all_indices = np.concatenate(([0], change_indices, [len(arr)]))
		run_lengths = np.diff(all_indices)
		run_values = arr[all_indices[:-1]]
		for n, (k, g) in enumerate(groupby(arr.tolist())):
			if n >= 3:
				return None # raise NotBorder()
			if k == border_color.value and len(list(g)) >= border_len:
				return True
		return False'''

	y = -1
	# border_lines = []
	for n, y in enumerate(range(image.shape[0])):
		is_border = get_border_or_bg(y)
		if is_border:  # color == border_color:
			yield n
			# border_lines.append(n)
	# return border_lines

@dataclass
class ImageAreaParam(TOMLDataclass):
	ypos: int = 0
	height: int = -1
	xpos: int = 0
	width: int = -1

@dataclass
class SplitImageAreaParam(TOMLDataclass):
	'''dual column layout'''
	ypos: int = 0
	height: int = -1
	start_xpos: int = 0
	end_xpos: int = -1

class TaimeeFilter:
	THRESHOLD = 237
	def __init__(self, given_image: np.ndarray | Path | str, params: dict[ImageFilterParam, int] = {}):
		self.org_image = given_image if isinstance(given_image, np.ndarray) else cv2.imread(str(given_image))
		self.params = params
		self.borders = []
		self.bin_image = cv2.threshold(self.org_image, self.THRESHOLD, 255, cv2.THRESH_BINARY)[1]
		self.non_nearby_borders = NonNearbyElems(thresh=self.bin_image.shape[0] // 20)
		for border in find_horizontal_borders(self.bin_image, border_color=BorderColor.BLACK):
			self.non_nearby_borders.add(border)
			self.borders.append(border)
		self.non_nearby_array = np.array(self.non_nearby_borders.elems)
		self.leading_y = self.borders[0] if len(self.non_nearby_borders.elems) == 4 else -1
		self.border_array = np.array(self.borders)
		if self.leading_y > 0:
			self.border_array = self.border_array - self.leading_y
		if len(self.non_nearby_borders.elems) == 4:
			self.non_nearby_array = self.non_nearby_array[1:] - self.leading_y

		
	def extract_heading(self, params: dict[ImageFilterParam, tuple[int, int]] | None = None, seek_bottom_shape: bool = False) -> ImageAreaParam:# tuple[int, int, int]:
		'''Return: (ypos, height, x_start)
		Use with self.leading_y like image[self.leading_y: self.leading_y + height, x_start:]'''
		if len(self.non_nearby_array) == 0:
			raise ValueError("No nearby borders found")
		if self.leading_y >= self.non_nearby_array[0]:
			raise ValueError("Leading y is not less than non-nearby array first element")
		heading_area = self.bin_image[self.leading_y:self.non_nearby_array[0] + self.leading_y, :].copy()
		for y in self.border_array:
			if y >= heading_area.shape[0]:
				break
			heading_area[y, :] = 255
		xpos = self.get_heading_avatar_end_xpos(heading_area, remove_borders=False)
		# scan from bottom
		if seek_bottom_shape:
			heading_h, heading_w = heading_area.shape[:2]
			shape_hlen = heading_w // 3
			def get_shape_hline(y: int):
				for n, (k, g) in enumerate(groupby(heading_area[y, :].tolist())):
					if n == 3 and k == 0 and len(list(g)) >= shape_hlen:
						return True
			shape_found = False
			for y in range(heading_h - 1, 0, -1):
				if get_shape_hline(y):
					shape_found = True
					break
			if not shape_found:
				raise ValueError("No shape found in heading bottom area!")
			heading_area[0, :] = 255
			y2 = -1
			bg_found = False
			for y2 in range(y, 0, -1):
				if np.all(heading_area[y2, xpos:] == 255):
					bg_found = True
					break
			if not bg_found:
				raise ValueError("No valid shape found (2)")
			'''cv2.imshow("Heading area bottom shape", heading_area[y2+1:y+1, xpos:])	
			cv2.waitKey(0)'''
			shape_area = heading_area[y2+1:y+1, xpos:]
			shape_w = shape_area.shape[1]
			black_found = False
			for x in range(shape_w):
				if np.any(shape_area[:, x] != 255):
					black_found = True
					break
			if not black_found:
				raise ValueError("No valid shape found (3)")
			bg_found = False
			for x2 in range(x, shape_w):
				if np.all(shape_area[:, x2] == 255):
					bg_found = True
					break
			if not bg_found:
				raise ValueError("No valid shape found (4)")
			print(f"Shape's rectangle size is:: height: {y - y2}, width: {x2 - x} ")
			print(f"and it's position is:: from heading area top: {y2}, from left: {xpos + x}")
			h_area_copy = heading_area.copy()
			cv2.rectangle(h_area_copy, (xpos + x, y2), (xpos + x2, y), (0, 0, 0), 2)
			cv2.imshow("Heading area bottom shape", h_area_copy)	
			cv2.waitKey(0)

		new_params = ImageAreaParam(0, y2, xpos)#, -1)
		if params is not None:
			params[ImageFilterParam.heading] = new_params
		return new_params

		

		
	@classmethod
	def get_heading_avatar_end_xpos(cls, image: np.ndarray, min_width: int = 8, 		borders: list[int] | None =None, remove_borders: bool = False)->int:
		height, width = image.shape[:2]

		# Remove horizontal borders from the image to simplify processing
		if remove_borders:
			image = image.copy()
			if borders is None:
				for y in (find_horizontal_borders(image)):
					image[y, :] = 255
			else:
				for y in borders:
					image[y, :] = 255

		px = -1
		for px in range(width):
			v_line = image[:, px]
			if np.any(v_line == 0):
				break
		if px == width:
			raise ValueError("Not enough valid width(%d) for the heading!" % width)
		# 2 pass the shape
		x = -1
		for x in range(px + 1, width):
			v_line = image[:, x]
			if np.all(v_line == 255):
				break
		if x < min_width:
			raise ValueError("Not enough valid width(%d) for the heading!" % x)
		return x
from dataclasses import field
@dataclass
class BinaryImage:
	given_image: ndarray | Path | str
	thresh_type: int = cv2.THRESH_BINARY # cv2.THRESH_OTSU
	thresh_value: float = 150.0
	single: bool = False
	cvt_color: int = cv2.COLOR_BGR2GRAY
	image_dict: dict[ImageDictKey, np.ndarray] | None = field(default_factory=dict)
	image_filter_params: dict[ImageFilterParam, int] = field(default_factory=dict)
	b_thresh_val: float = 235.0
	binarize: bool = True
	image_filter_areas: ImageFilterAreas | None = None

	def __post_init__(self):
		org_image = image_fullpath = None
		match self.given_image:
			case ndarray():
				org_image = self.given_image
			case Path():
				image_fullpath = str(self.given_image.resolve())
			case str():
				image_fullpath = str(Path(self.given_image).resolve())
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
		self.mono_image = cv2.cvtColor(org_image, self.cvt_color)

	def bin_image(self, thresh_val: float | None = None):
			return cv2.threshold(
				self.mono_image, thresh=thresh_val if thresh_val is not None else self.b_thresh_val, maxval=255, type=self.thresh_type
			)[1]
	'''
		else:
			self.auto_thresh_val = 0
			self.bin_image = self.mono_image'''


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
	image_filter_areas: ImageFilterAreas | None = None,
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
	non_nearby_elems = NonNearbyElems(thresh=height // 20)
	horizontal_borders = []
	try:
		for y in find_horizontal_borders(b_image):
			non_nearby_elems.add(y)
			horizontal_borders.append(y)
	except NoBorderError:
		raise ValueError("No horizontal borders found!")
	non_nearby_elems_array = np.array(non_nearby_elems.elems)
	leading_height = non_nearby_elems_array[0]
	non_nearby_elems_array[:] -= leading_height
	non_nearby_elems = non_nearby_elems_array[1:].tolist()
	heading_elem = non_nearby_elems[0]
	# from image_filter import TaimeeFilter
	taimee_filter = TaimeeFilter(b_image, horizontal_borders)
	heading_ypos, heading_height, heading_xpos = taimee_filter.extract_heading()
	# heading_area_xpos = TaimeeFilter.get_heading_avatar_end_xpos(b_image[leading_height:heading_elem+leading_height, :], borders=horizontal_borders)
	ocr = TesseractOCR()
	# from pandas import DataFrame
	from pytesseract import Output as TesseractOutput
	ocr_dataframe = ocr.exec_ocr(mono_image[taimee_filter.leading_y: taimee_filter.leading_y+heading_height, heading_xpos:], output_type=TesseractOutput.DATAFRAME)
	heading_text = ocr_dataframe[ocr_dataframe['conf'] > 0]['text']	
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
	heading_area: HeadingArea = trim_heading(pre_h_image, image_filter_params)#, return_as_cuts=True)
	h_image = pre_h_image[:heading_area.height, heading_area.xpos:]
	h_image2 = heading_area.crop_image(pre_h_image)
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
		image_dict[ImageDictKey.work_time] = image[ypos_list[0] : ypos_list[1], :]
		image_dict[ImageDictKey.shift_start] = image[ypos_list[0] : ypos_list[1], :xpos]
		image_dict[ImageDictKey.shift_end] = image[
			ypos_list[0] : ypos_list[1], xpos2:
		]
		image_dict[ImageDictKey.break_time] = image[ypos_list[1] : ypos_list[-1], :]
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




def find_border(
	image: np.ndarray,
	border_color: BorderColor = BorderColor.BLACK,
	edge_ratio: float = 0.10,
) -> tuple[int, int]:
	"""Find border of the image.
	Return: border_ypos, border_len"""
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

	def get_border_or_bg(y: int) -> bool:  # | None:
			# Returns True if border is found, False if background is found, raises NotBorder if not found
			for n, (k, g) in enumerate(groupby(image[y, :])):
				if n >= 3:
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
	# return_as_cuts: bool = False,
) -> HeadingArea:
	"""h_image: binarized i.e. 0 or 255
	background is 255
	Return: trimmed heading or list of trimmed headings(return_as_cuts=True) as HeadingCuts(bottom_cut_height, left_cut_width)
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
	return HeadingArea(ypos=0, height=heading_height, xpos=left_pad) # if return_as_cuts else h_image[:heading_height, left_pad:]


def get_split_shifts(
	image: np.ndarray, params: dict[ImageFilterParam, int] = {}, set_params=True, 
return_as_cuts: bool = False, center_rate = 0.5
) -> tuple[np.ndarray, np.ndarray] | ShiftArea:
	"""Split image into left and right;black-filled shape's x position is center_rate;
	Args: h_image: binarized i.e. 0 or 255
	background is 255(white);
	Return: tuple[left_width, right_start_xpos] if return_as_cuts else tuple[left_image, right_image]
	"""
	if not(0 < center_rate < 1):
		raise ValueError("center_rate must be between 0 and 1!")
	width = image.shape[1]
	center = int(width * center_rate)
	x = -1
	try:
		x = params[ImageFilterParam.shift_from_width]
	except KeyError:
		for x in range(center - 1, 0, -1):
			v_line = image[:, x]
			if np.all(v_line == 255):
				break
	if x < 0:
		raise ValueError("Not enough valid width(%d) for the shift!" % x)
	left_area_width = x  # + 1
	x = -1
	try:
		x = params[ImageFilterParam.shift_until_xpos]
	except KeyError:
		for x in range(center + 1, image.shape[1]):
			v_line = image[:, x]
			if np.all(v_line == 255):
				break
	if x < 0:
		raise ValueError("Not enough valid width(%d) for the shift!" % x)
	right_area_xpos = x  # + 1
	if set_params:
		params[ImageFilterParam.shift_from_width] = left_area_width
		params[ImageFilterParam.shift_until_xpos] = right_area_xpos
	return ShiftArea(ypos=0, height=image.shape[0], start_width=left_area_width, end_xpos=right_area_xpos) if return_as_cuts else (image[:, :left_area_width], image[:, right_area_xpos:])



def merge_nearby_elems(elems: Sequence[int], thresh=9) -> Iterator[int]:
	if len(elems) < 2:
		raise ValueError("elems must be 2 or more!")
	elem0 = elems[0]
	elem = sent = None
	for elem in elems[1:]:
		sent = False
		if elem - elem0 > thresh:
			yield elem0
			sent = True
			elem0 = elem
	if sent:
		yield elem0


