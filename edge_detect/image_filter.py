from io import IOBase
from enum import IntEnum, auto
from dataclasses import field
from collections import deque
import matplotlib.pyplot as plt
from cv2 import UMat
import cv2
from tesseract_ocr import TesseractOCR

# from cvc2.typing import MatLike
# import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from pathlib import Path
from typing import Iterator, Sequence, NamedTuple, Optional, Sequence
from dataclasses import dataclass
from enum import Enum
import sys
import matplotlib.pyplot as plt
from inspect import isclass
from itertools import groupby
from fancy_dataclass import TOMLDataclass

from near_bunch import NearBunch, NearBunchException, NoBunchException

cwd = Path(__file__).resolve().parent
sys.path.append(str(cwd.parent))
from set_logger import set_logger

logger = set_logger(__name__)

class ImageFilterException(Exception):
	pass

class NotEnoughBordersException(ImageFilterException):
	pass

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
	y_offset = (KeyUnit.PIXEL, 1)
	heading_button = (KeyUnit.TEXT, 0)  # heading"
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
class ImageAreaParam(TOMLDataclass):
	y_offset: int = 0
	height: int | None = None
	x_offset: int = 0
	width: int | None = None

	# param: ItemAreaParam # NamedTuple:read only

	def __post_init__(self):
		if self.x_offset < 0:
			raise InvalidValueError("x offset must be positive")
		if self.height is not None:
			if self.height != -1 and self.height <= 0:
				raise InvalidValueError("height must be larger than 0 except None or -1")
		if self.width is not None:
			if self.width != -1 and self.width <= 0:
				raise InvalidValueError("width must be larger than 0 except None or -1")

	@classmethod
	def from_image(cls, image: np.ndarray, offset:int=0) -> "ImageAreaParam":
		height, width = image.shape[:2]
		return cls(height=height, width=width)

	@property
	def param(self)-> Int4:
		return (self.y_offset, self.height or -1, self.x_offset, self.width or -1)


	def as_slice_param(self) -> Sequence[Int4]:
		return ((self.y_offset, (self.y_offset + self.height) if self.height else -1, self.x_offset, (self.x_offset + self.width) if self.width else -1),)

	def crop_image(self, image: np.ndarray) -> Iterator[np.ndarray]:
		for param in self.as_slice_param():
			yield image[param[0]:param[1], param[2]:param[3]]


@dataclass
class OffsetArea:
	height: int

class XOffsetHeight(NamedTuple):
	x_offset: int
	height: int

@dataclass
class XYOffset:
	x: int
	y: int
	@property
	def from_bottom(self)->int:
		return -self.y
	@property
	def from_left(self)->int:
		return self.x

class XYRange(NamedTuple):
	y: range
	x: range

class FromBottomLabelRange(NamedTuple):
	from_bottom: int
	label_range: range

class XYPosition(Enum):
	OFFSET = XYOffset
	RANGE = XYRange

class FigurePart(Enum):
	AVATAR = auto() # like (figure)
	LABEL = auto() # like [string]

@dataclass
class HeadingAreaParam(ImageAreaParam):
	'''
	Only this area the y_offset value is negative. It means to trim the bottom part i.e.: image[0:(image.shape[0] + y_offset)].
	Necessary named parameters: height, xpos'''

	@classmethod
	def get_label_text_start(cls) -> str:
		return "LABEL TEXT START"



	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(self.as_toml() + '\n')

	def as_toml(self, **kwargs):
		name = kwargs.get('name', '').strip()
		name_str = f".{name}" if name else ""
		return f"{self.__class__.__name__}{name_str} = {str(list(self.param))}"

@dataclass
class TaimeeHeadingAreaParam(HeadingAreaParam):
	@classmethod
	def get_label_text_start(cls) -> str:
		return 'この店'

	@classmethod
	def from_image(cls, image: np.ndarray, offset: int = 0, figure_parts: dict[
	FigurePart, XYRange|FromBottomLabelRange] = {}, image_check=False, label_check=True) -> "TaimeeHeadingAreaParam":
		xy_offset = cls.check_image(image, figure_parts=figure_parts, image_check=image_check, label_check=label_check)
		return cls(y_offset=xy_offset.y, x_offset=xy_offset.x)

	@classmethod
	def crop_image(cls, image: np.ndarray, from_bottom: int, x_offset: int) -> np.ndarray:
		return image[0:image.shape[0] - abs(from_bottom), x_offset:]

	@classmethod
	def scan_image_range_x(cls, image: np.ndarray)-> range:
		'''rt = stop = -1
		for i, e in enumerate(np.argmax(image[:, :],axis=0)):
			if start == -1:
				if e == 0: # black
					start = i
			else:
				if e != 0: # white
					stop = i
					break'''
			
		x = -1
		black_found = False
		for x in range(image.shape[1]):
			if np.any(image[:, x] == 0):
				black_found = True
				break
		if not black_found:
			raise ValueError("No black found in scan area!")
		x0 = x
		white_found = False
		for x in range(x0, image.shape[1]):
			if np.all(image[:, x] == 255):
				white_found = True
				break
		if not white_found:
			raise ValueError("No white found in scan area!")		
		return range(x0, x)

	@classmethod
	def scan_image_range_y(cls, scan_area: np.ndarray)-> range:
		## scan horizontal lines to find the vertical range of the shape
		y = -1
		black_found = False
		for y in range(scan_area.shape[0]):
			if np.any(scan_area[y, :] == 0):
				black_found = True
				break
		if not black_found:
			raise ValueError("No black found in scan area(2)!")
		y0 = y
		white_found = False
		for y in range(y0, scan_area.shape[0]):
			if np.all(scan_area[y, :] == 255):
				white_found = True
				break
		if not white_found:
			raise ValueError("No white found in scan area(2)!")
		return range(y0, y)

	@classmethod
	def image_shape_check_circle(cls, shape_area: np.ndarray, diff_ratio:float=0.1) -> None:
		canvas = np.full(shape_area.shape[:2], 255, np.uint8)
		cv2.circle(canvas, (shape_area.shape[1]//2, shape_area.shape[0]//2 - 1), shape_area.shape[1]//2, 0, -1)
		# compare shape_area with circle edges(left ang right)
		abs_diff_list = []
		for line in range(shape_area.shape[0]):
			v = shape_area[line, :]
			black_pos = np.where(v == 0)
			if black_pos[0].size > 1:
				l_diff = shape_area[line, black_pos[0][0]] - canvas[line, black_pos[0][0]]
				r_diff = shape_area[line, black_pos[0][-1]] - canvas[line, black_pos[0][-1]]
				abs_diff_list.append(abs(l_diff)+ abs(r_diff))
		abs_diff_sum = sum(abs_diff_list)
		abs_diff_sum_ratio = abs_diff_sum / shape_area.size
		if abs_diff_sum_ratio > diff_ratio:
			raise ValueError("Detected avatar shape is too different from circle!")

	@classmethod
	def check_image(cls, image: np.ndarray, image_check=False, figure_parts: dict[
	FigurePart, XYRange|FromBottomLabelRange] = {},
	avatar_shape_check=False, label_check=True) -> XYOffset:
		'''
		avatar_area | label_area: (y_range, x_range), (from_bottom, from_left)]
		If avatar_area or label_area was/were given as empty list, it/they are fulfilled with found ones.
		returns (x_offset, y_trim) for heading_area should offset x and trim height: y_trim is a negative value, so, adding it to image height'''
		# check if avatar_area is not None
		'''if avatar_area:
			if len(avatar_area) != 2:
				raise ValueError("avatar_area must be a list of 2 elements")
			if not (XYOffset not in avatar_area or XYRange not in avatar_area):
				raise ValueError("avatar area must contain FromBottomFromLeft and XYRange")
			for item in avatar_area:
				if isinstance(item, XYOffset):
					avatar_from_bottom_from_left = item
				else:
					avatar_x_y_range = item'''
		# check if avatar circle at the left side of the area between the borders(1st and 2nd)
		## scan vertical lines to find the horizontal range of the shape (expetcing as a circle)
		if FigurePart.AVATAR not in figure_parts:
			x_range = cls.scan_image_range_x(image)
			y_range = cls.scan_image_range_y(image[:, x_range.start:x_range.stop])
			figure_parts[FigurePart.AVATAR] = XYRange(y=y_range, x=x_range)

		if avatar_shape_check:
			# check black pixel in the shape area
			y_range = figure_parts[FigurePart.AVATAR].y
			x_range = figure_parts[FigurePart.AVATAR].x
			shape_area = image[y_range.start:y_range.stop, x_range.start:x_range.stop] #scan_area[v_range[0]:v_range[1], h_range[0]:h_range[1]]
			cls.image_shape_check_circle(shape_area)

		# shape_area_copy_as_white = np.full(shape_area.shape, 255, np.uint8)
		# cv2.circle(shape_area_copy_as_white, (shape_area.shape[1]//2, shape_area.shape[0]//2), shape_area.shape[1]//2, 0, -1)
		# diff_image = cv2.bitwise_xor(canvas, shape_area_copy_as_white)
		### draw a circle on virtual_circle_area
		'''SUBPLOT_SIZE = 3
		fig, ax = plt.subplots(SUBPLOT_SIZE, 1)#, figsize=(10, 4*SUBPLOT_SIZE))
		for r in range(SUBPLOT_SIZE):
			ax[r].invert_yaxis()
			ax[r].xaxis.tick_top()
			ax[r].set_title(f"Row {r+1}")
		ax[0].imshow(canvas, cmap='gray')
		ax[1].imshow(shape_area, cmap='gray')
		ax[2].imshow(abs_diff_list, cmap='gray')
		plt.show()
		diff_white_count = np.count_nonzero(diff_image != 0)
		diff_white_percentage = diff_white_count * 100 / diff_image.size
		if diff_white_percentage > 5:
			raise ValueError("Detected avatar circle area white diff is too large!")
		shape_area_black_count = np.count_nonzero(shape_area == 0)
		if abs_diff_sum / shape_area_black_count > 0.1:
			raise ValueError("Detected avatar circle area black diff is too large!")
		'''
		if FigurePart.LABEL not in figure_parts:
			# scan label-like shape from bottom, right side of the avatar area
			start_x = figure_parts[FigurePart.AVATAR].x.stop
			scan_area = image[:, start_x:]
			if image_check:
				cv2.imshow("scan_area", scan_area)
				cv2.waitKey(0)
			scan_h, scan_w = scan_area.shape[:2]
			label_w_min = scan_w // 4
			def get_run_length(line: Sequence[int]):
				for n, (k, g) in enumerate(groupby(line)):
					if n == 0:
						_g = g
					elif n == 1 and k == 0 and (w:=len(list(g))) >= label_w_min:
						return _g, w

			label_bottom_line = None
			for y in range(scan_h - 1, 0, -1):
				if (g_w := get_run_length(scan_area[y, :].tolist())):
					label_bottom_line = g_w[1]
					break
			if not label_bottom_line:
				raise ValueError("No label-like shape's bottom line found in heading bottom area!")
			label_bottom_x_offset = len(list(g_w[0]))
			# scan_area[0, :] = 255
			y2 = -1
			bg_found = False
			x = figure_parts[FigurePart.AVATAR].x.stop
			for y2 in range(y, 0, -1):
				if np.all(scan_area[y2, x:] == 255):
					bg_found = True
					break
			if not bg_found:
				raise ValueError("No b.g. above label-like shape!")
			from_bottom_label_range = FromBottomLabelRange(y2 - scan_area.shape[0], range(start_x + label_bottom_x_offset, start_x + label_bottom_x_offset + label_bottom_line))
			if label_check:
				from tesseract_ocr import TesseractOCR, Output
				ocr = TesseractOCR()
				ocr_area = image[image.shape[0] + from_bottom_label_range.from_bottom:, start_x:]
				ocr_result = ocr.exec(ocr_area, output_type=Output.DATAFRAME, psm=7)
				label_text = ''.join(list(ocr_result[ocr_result['conf'] > 50]['text']))
				if not label_text.startswith(
				lts:=cls.get_label_text_start()):
					raise ValueError("Improper text start: " + lts)
			figure_parts[FigurePart.LABEL] = from_bottom_label_range
		return XYOffset(x=figure_parts[FigurePart.AVATAR].x.stop, y=figure_parts[FigurePart.LABEL].from_bottom)

@dataclass
class ShiftAreaParam(ImageAreaParam):
	'''Needs to initialize using named parameters::
	start_width: as xpos
	end_xpos: as width
	while start_xpos is 0 and end_width is -1
	'''
	@classmethod
	def check_image(cls, image: np.ndarray, image_check=False)-> tuple[int, int]:
		'''returns (left_end, right_start)'''
		# check if avatar circle at the left side of the area between the borders(1st and 2nd)
		## scan vertical lines to find the horizontal range		 of the shape (expetcing as a black filled rectangle of left side flat)

		x = -1
		all_white = False
		for x in range(image.shape[1] // 2 - 1, 0, -1):
			if np.all(image[:, x] != 0):
				all_white = True
				break
		if not all_white:
			raise ValueError("No all white found in left half of scan area!")
		x0 = x
		all_white = False
		for x in range(image.shape[1] // 2, image.shape[1]):
			if np.all(image[:, x] != 0):
				all_white = True
				break
		if not all_white:
			raise ValueError("No all white found in right half of scan area!")
		if image_check:
			cv2.imshow("				image", image[:, x0:x])
			cv2.waitKey(0)
		return x0, x

	@classmethod
	def from_image(cls, image: np.ndarray, offset: int, image_check=False) -> "ShiftAreaParam":
		left, right = cls.check_image(image, image_check)
		return cls(ypos=offset, height=image.shape[0], xpos=left, width=right)

	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(self.as_toml() + '\n')

	def as_toml(self, **kwargs):
		name = kwargs.get('name', '').strip()
		name_str = f".{name}" if name else ""
		return f"{self.__class__.__name__}{name_str} = {str(list(self.param))}"
	@property
	def start_width(self)-> int: # start-from time
		return self.x_offset

	@property
	def end_xpos(self)-> int: # end-by time
		return self.width
@dataclass		
class ShiftStartAreaParam(ImageAreaParam):
	pass

@dataclass
class ShiftEndAreaParam(ImageAreaParam):
	pass

	def crop_image(self, image: np.ndarray) -> Iterator[np.ndarray]:
		for param in self.as_slice_param():
			yield image[param[0]:param[1], param[2]:param[3]]

@dataclass
class BreaktimeAreaParam(ImageAreaParam):
	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(self.as_toml() + '\n')

	def as_toml(self, **kwargs):
		name = kwargs.get('name', '').strip()
		name_str = f".{name}" if name else ""
		return f"{self.__class__.__name__}{name_str} = {str(list(self.param))}"

@dataclass
class PaystubAreaParam(ImageAreaParam):
	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(self.as_toml() + '\n')

	def as_toml(self, **kwargs):
		name = kwargs.get('name', '').strip()
		name_str = f".{name}" if name else ""
		return f"{self.__class__.__name__}{name_str} = {str(list(self.param))}"
@dataclass
class SalaryAreaParam(ImageAreaParam):
	def as_toml(self, **kwargs):
		name = kwargs.get('name', '').strip()
		name_str = f".{name}" if name else ""
		return f"{self.__class__.__name__}{name_str} = {str(list(self.param))}"
	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(self.as_toml() + '\n')


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

	area_key_list = [ImageDictKey.y_offset, ImageDictKey.heading, ImageDictKey.shift_start, ImageDictKey.shift_end, ImageDictKey.break_time, ImageDictKey.payslip, ImageDictKey.salary]

	areas = {
		ImageDictKey.y_offset: OffsetArea,
		ImageDictKey.heading: HeadingAreaParam,
		ImageDictKey.shift_start: ShiftStartAreaParam,
		ImageDictKey.shift_end: ShiftEndAreaParam,
		ImageDictKey.break_time: BreaktimeAreaParam,
		ImageDictKey.payslip: PaystubAreaParam,
		ImageDictKey.salary: SalaryAreaParam,
	}
	heading: HeadingAreaParam # midashi
	shift: ShiftAreaParam # syuugyou jikan
	break_time: BreaktimeAreaParam # kyuukei jikan
	paystub: PaystubAreaParam # meisai
	salary: SalaryAreaParam # kyuuyo
	y_offset: int = 0

class ImageAreaName(Enum):
	heading = HeadingAreaParam
	shift = ShiftAreaParam
	breaktime = BreaktimeAreaParam
	paystub = PaystubAreaParam
	salary = SalaryAreaParam

class ImageFilterParam(Enum):
	y_offset = 0, 0
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


class DistantElems:
	def __init__(self,
		distance: int = 10,
		elems: list[int] = [],
		excluded: list[int] = [],
	):
		self.distance = distance
		self.elems = elems
		self.excluded = excluded

	def add(self, i: int) -> int:
		'''returns 1 if added'''
		if len(self.elems) == 0:
			self.elems.append(i)
			return 1
		else:
			if i - self.elems[-1] > self.distance:
				self.elems.append(i)
				return 1
			else:
				self.excluded.append(i)
				return 0





class BorderColor(Enum):
	WHITE = 255
	BLACK = 0


class NoBorderError(ImageFilterException):
	pass

def find_horizontal_borders(
	image: np.ndarray,
	border_color: BorderColor = BorderColor.BLACK,
	edge_ratio: float = 0.10,
	offset: int = 0
) -> Iterator[int]:
	"""Find border lines in the image.
	Return: list of border ypos"""
	if offset < 0 or offset >= image.shape[0] - 1:
		raise ValueError("Offset is out of range!")


	edge_len = int(image.shape[1] * edge_ratio)
	border_len = image.shape[1] - edge_len * 2


	def get_border_or_bg(y: int) -> bool | None:
		# Returns True if border is found, False if background is found, else returns None
		if np.all(image[		y + offset, :] == 255):
			return False
		arr = image[y, edge_len:-edge_len]
		if np.all(arr == border_color.value):
			return True
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
	for y in (range(image.shape[0] - offset)):
		is_border = get_border_or_bg(y)
		if is_border:  # color == border_color:
			yield y
			# border_lines.append(n)
	# return border_lines



@dataclass
class SplitImageAreaParam(TOMLDataclass):
	'''dual column layout'''
	ypos: int = 0
	height: int = -1
	start_xpos: int = 0
	end_xpos: int = -1

@dataclass
class AreaBeginEnd:
	start: int
	height: int
	@property
	def begin(self):
		return self.start
	@property
	def end(self):
		return self.start + self.height

class OffsetException(Exception):
	pass

class OffsetInt:
	def __init__(self, value: int, limit: int):
		self.value = value
		self.limit = limit
		if value >= limit:
			raise OffsetException("Initial offset is out of range!")

	def inc(self, value=1):
		if self.value + value >= self.limit:
			raise OffsetException("Offset is out of range!")
		self.value += value

	def set(self, value: int):
		if value >= self.limit:
			raise OffsetException("Unable to set: value is out of range!")
		self.value = value

def _plot(images: Sequence[np.ndarray]):
		SUBPLOT_SIZE = len(images)
		if SUBPLOT_SIZE == 1:
			SUBPLOT_SIZE += 1
		fig, ax = plt.subplots(1, SUBPLOT_SIZE)
		for r in range(SUBPLOT_SIZE):
			ax[r].invert_yaxis()
			ax[r].xaxis.tick_top()
			ax[r].set_title(f"Row {r+1}")
		for n, image in enumerate(images):
			ax[n].imshow(image)
		plt.show()

class TaimeeFilter:
	THRESHOLD = 237
	BORDERS_MIN = 3
	BORDERS_MAX = 4
	LABEL_TEXT_START = 'この店'

	def __init__(self, image: np.ndarray | Path | str, param_dict: dict[ImageAreaName, ImageFilterParam] = {}, show_check=False):
		self.image = image if isinstance(image, np.ndarray) else cv2.imread(str(image))
		if self.image is None:
			raise ValueError("Failed to load image")
		self.params = param_dict
		bin_image = cv2.threshold(self.image, self.THRESHOLD, 255, cv2.THRESH_BINARY)[1]

		# find borders as bunches
		border_offset_list: deque[tuple[int, int]] = deque()
		# border_offset_list: deque[BorderOffset] = deque()

		# _border_offset_list: list[BorderOffset] = []
		'''with-margin / without-margin
			margin
		 0 HL
			header
		 1 HL
			shift
		 2 HL
			breaktime
		 3 HL
		'''
		# last_offset = 0
		n = -1
		for n, (b, o) in enumerate(get_horizontal_border_bunches(bin_image, min_bunch=3, offset_list=border_offset_list)):
			# if n == 0: y_margin = b.elems[-1] + 1
					# last_b_end = b.elems[-1]
				# elif n == 1: horizontal_border_offset_list.append(BunchOffset(b, b.elems[0] - 1))
			if n == 3:
				break
				# last_offset += b.elems[-1] + 1


		margin_area = border_offset_list.popleft()
		y_margin = border_offset_list[0][0]
		bin_image = bin_image[y_margin:, :]
		border_offsets = np.array(border_offset_list)
		border_offsets -= y_margin # border_offsets[0][0]
		if __debug__:
			canvas = bin_image.copy()
			canvas[:, 0:4] = 255
			for n, (start, stop) in enumerate(border_offsets):
				canvas[start:stop, n] = 0
			_plot([canvas])
		self.y_margin = y_margin

		self.y_origin = y_origin = border_offsets[0][1]
		heading_area_figure_parts = {}
		try:
			heading_area_param = TaimeeHeadingAreaParam(*param_dict[ImageAreaName.heading])
		except (KeyError, TypeError):
			heading_area_param = TaimeeHeadingAreaParam.from_image(bin_image[:self.y_origin, :], figure_parts=heading_area_figure_parts)
			if show_check:
				SUBPLOT_SIZE = 2
				fig, ax = plt.subplots(SUBPLOT_SIZE, 1)
				for r in range(SUBPLOT_SIZE):
					ax[r].invert_yaxis()
					ax[r].xaxis.tick_top()
					ax[r].set_title(f"Row {r+1}")
				ax[0].imshow(TaimeeHeadingAreaParam.crop_image(bin_image[:y_origin, :],
				from_bottom=heading_area_param.y_offset, x_offset=heading_area_param.x_offset), cmap='gray')
				plt.show()
		if show_check:
			show_image = self.image[self.y_origin:self.y_origin + heading_area_param.y_offset, heading_area_param.x_offset:]
			do_show_check("heading_area", heading_area_param, show_image)
		self.area_param_list: list[ImageAreaParam] = [heading_area_param]
		# get shift area
		try:
			area_param = ShiftAreaParam(*param_dict[ImageAreaName.shift])
		except (KeyError, TypeError):
			area_top = border_offset_list[0].elems[-1] + 1
			y_origin = border_offset_list[1].elems[0]
			scan_image = bin_image[area_top:y_origin, :]
			area_param = ShiftAreaParam.from_image(scan_image, offset=area_top, image_check=show_check)
		if show_check:
			area_image = bin_image[area_param.y_offset:area_param.y_offset + area_param.height, :]
			do_show_check("shift_area", area_param, area_image)
		self.area_param_list.append(area_param)
		# get breaktime area
		try:
			area_param = BreaktimeAreaParam(*param_dict[ImageAreaName.breaktime])
		except (KeyError, TypeError):
			area_top = border_offset_list[1].elems[-1] + 1
			y_origin = border_offset_list[2].elems[0]
			area_param = BreaktimeAreaParam(y_offset=area_top, height=y_origin - area_top)
		if show_check:
			area_image = bin_image[area_param.y_offset:area_param.y_offset + area_param.height, :]
			do_show_check("breaktime area", area_param, area_image)
		self.area_param_list.append(area_param)
		# get paystub area
		try:
			area_param = PaystubAreaParam(*param_dict[ImageAreaName.paystub])
		except (KeyError, TypeError):
			area_top = border_offset_list[2].elems[-1] + 1
			area_param = BreaktimeAreaParam(y_offset=area_top)
		if show_check:
			area_image = bin_image[area_param.y_offset:, :]
			do_show_check("paystub area", area_param, area_image)
		self.area_param_list.append(area_param)
		if show_check:
			cv2.destroyAllWindows()
		
	def extract_heading(self, params: dict[ImageDictKey, tuple[int, int]] | None = None, seek_button_shape: bool = False, button_text: Optional[list[str]] = None) -> ImageAreaParam|tuple[ImageAreaParam, str]:
		'''Return: (ypos, height, x_start)
		Use with self.y_offset like image[self.y_offset: self.y_offset + height, x_start:]'''
		if len(self.distant_array) == 0:
			raise ValueError("No nearby borders found")
		if self.y_origin >= self.distant_array[0]:
			raise ValueError("Y offset is not less than non-nearby array first element")
		heading_area = bin_image[self.y_origin:self.distant_array[0] + self.y_origin, :].copy()
		for y in self.horizontal_lines:
			if y >= heading_area.shape[0]:
				break
			heading_area[y, :] = 255
		xpos = self.get_heading_avatar_end_xpos(heading_area, remove_borders=False)
		# scan button like shape from bottom
		if seek_button_shape or (button_text is not None):
			heading_h, heading_w = heading_area.shape[:2]
			button_w_min = heading_w // 3
			def get_line(line: Sequence[int]):
				for (k, g) in groupby(line):
					if k == 0 and (w:=len(list(g))) >= button_w_min:
						return w
			def get_hline(y: int):
				for n, (k, g) in enumerate(groupby(heading_area[y, :].tolist())):
					if n == 3 and k == 0 and (w:=len(list(g))) >= button_w_min:
						return w
			button_bottom_line = None
			for y in range(heading_h - 1, 0, -1):
				if (w:=get_line(heading_area[y, :].tolist())):
					button_bottom_line = w
					break
			if not button_bottom_line:
				raise ValueError("No button found in heading bottom area!")
			heading_area[0, :] = 255
			y2 = -1
			bg_found = False
			for y2 in range(y, 0, -1):
				if np.all(heading_area[y2, xpos:] == 255):
					bg_found = True
					break
			if not bg_found:
				raise ValueError("No bg above button shape!")
			button_area = heading_area[y2:y, xpos:]
			cv2.imshow("Button area", button_area)
			cv2.waitKey(0)
			# find contours of the button
			contours, _ = cv2.findContours(button_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			contour = contours[0]
			x, y, w, h = cv2.boundingRect(contour)
			button_area_copy = button_area.copy()
			cv2.rectangle(button_area_copy, (x, y), (x + w, y + h), 0, 2)
			cv2.imshow("Button area and its bounding rect", button_area_copy)
			cv2.waitKey(0)
			print(f"Bottom Shape's bounding rectangle size is:: height: {h}, width: {w} ")
			h_area_copy = heading_area.copy()
			cv2.rectangle(h_area_copy, (xpos + x, y2), (xpos + x2, y), (0, 0, 0), 2)
			cv2.imshow("Heading area bottom shape", h_area_copy)	
			cv2.waitKey(0)
			# button_text: str | None = None
			if button_text is not None:
				from tesseract_ocr import TesseractOCR, Output
				ocr = TesseractOCR()
				ocr_result = ocr.exec(heading_area[y2:y, xpos:], output_type=Output.DATAFRAME, psm=7)
				button_text += list(ocr_result[ocr_result['conf'] > 50]['text'])
				# print(f"OCR Result text: {ocr_text}")

			assert np.any(heading_area[y2 + 1, :] != 255)
			button_top_line = get_line(heading_area[y2 + 1,:].tolist()) # length
			if abs(button_top_line - button_bottom_line) > 10:
				raise ValueError(f"Button shape is not top-bottom symmetrical! top: {button_top_line}, bottom: {button_bottom_line}")	
			'''cv2.imshow("Heading area bottom shape", heading_area[y2+1:y+1, xpos:])	
			cv2.waitKey(0)'''
			# try to find button width
			button_band = heading_area[y2+1:y+1, xpos:]
			shape_w = button_band.shape[1]
			black_found = False
			for x in range(shape_w):
				if np.any(button_band[:, x] != 255):
					black_found = True
					break
			if not black_found:
				raise ValueError("No valid shape found (3)")
			bg_found = False
			for x2 in range(x, shape_w):
				if np.all(button_band[:, x2] == 255):
					bg_found = True
					break
			if not bg_found:
				raise ValueError("No valid shape found (4)")
			print(f"Button Shape's rectangle size is:: height: {y - y2}, width: {x2 - x} ")
			print(f"and it's position is:: from heading area top: {y2}, from left: {xpos + x}")
			assert (y + 2 ) > (y2 - 1)
			assert (x2 + 1) > (x - 1)
			button_rect = button_band[:, x:x2]
			# cv2.imshow("Heading shape area ", shape_band)	
			cv2.imshow("Heading area bottom shape rect", button_rect)	
			cv2.waitKey(0)


		new_params = ImageAreaParam(0, y2, xpos)#, -1)
		if params is not None:
			params[ImageDictKey.heading] = new_params
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
			bin_image = self.mono_image'''

# end def taimee

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
) -> HeadingAreaParam:
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
		env_filter_toml_path = dotenv_values(stream=f)[FILTER_TOML_PATH_STR]
		if env_filter_toml_path:
				filter_toml_path = env_filter_toml_path
	except KeyError:
		logger.warning("KeyError: '%s' not found in %s", FILTER_TOML_PATH_STR, ENV_FILE_NAME)
	except FileNotFoundError:
		logger.warning("FileNotFoundError: '%s' not found", ENV_FILE_NAME)
	if not filter_toml_path:
		raise ValueError("Error: failed to load 'filter_toml_path'!")
	try:
		with open(filter_toml_path, 'rb') as f:
		## skip bottom shape
			y = -1
			for y in range(dy, 0, -1):
				if np.all(h_image[y, left_pad:] == 255):
					break
			if y < min_height:
				raise ValueError("Not enough valid height(%d) for the heading!" % y)
			heading_height = y  # + 1
			params[ImageFilterParam.heading_height] = heading_height
	except FileNotFoundError:
		raise ValueError("Error: failed to load 'filter_toml_path'!")
	return HeadingAreaParam(y_offset=0, height=heading_height, x_offset=left_pad) # if return_as_cuts else h_image[:heading_height, left_pad:]


def get_split_shifts(
	image: np.ndarray, params: dict[ImageFilterParam, int] = {}, set_params=True, 
return_as_cuts: bool = False, center_rate = 0.5
) -> tuple[np.ndarray, np.ndarray] | ShiftAreaParam:
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
	return ShiftAreaParam(y_offset=0, height=image.shape[0], start_width=left_area_width, end_xpos=right_area_xpos) if return_as_cuts else (image[:, :left_area_width], image[:, right_area_xpos:])



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

class BorderOffset(NamedTuple):
	bunch: NearBunch
	offset: int

def get_horizontal_border_bunches(bin_image: np.ndarray, y_offset:int=0, bunch_thresh: int=10, min_bunch:int=3, max_bunch:int=10, offset_list: list[tuple[int, int]] | None = None) -> Iterator[BorderOffset]: # tuple[NearBunch, int]]:
	# bunches: list[NearBunch] = []
	offseter = OffsetInt(y_offset, limit=bin_image.shape[0])
	bunch: NearBunch | None = None
	last_offset: int = offseter.value
	range_start: int = -1
	for n in range(max_bunch):
		range_start = bunch.elems[-1] + 1 if bunch else 0
		range_start += last_offset
		try:
			last_offset = offseter.value
			bunch = find_horizontal_border_bunch(bin_image, bunch_thresh=bunch_thresh, y_offset=offseter)
			if offset_list is not None:
				range_stop = bunch.elems[0] + last_offset
				offset_list.append((range_start, range_stop))
		except	NoBunchException:
			if n < min_bunch:
				raise NotEnoughBordersException("Not enough borders found!")
			else:
				return # break
			
		# for n, e in enumerate(bunch.elems): bunch.elems[n] = e + y_offset
		yield BorderOffset(bunch, last_offset)
		# y_offset += bunch.elems[-1] + 1
	# return bunches

def find_horizontal_border_bunch(bin_image: np.ndarray, y_offset:OffsetInt|None=None, bunch_thresh: int=10) -> NearBunch:
	''' Increment Y_offset as bunch.elems[-1] + 1 '''
	bunch = NearBunch(bunch_thresh)
	for y in find_horizontal_borders(bin_image[y_offset.value if y_offset else 0:, :], border_color=BorderColor.BLACK):
		try:
			bunch.add(y)
		except NearBunchException:
			break
	if bunch.bunch_count == 0:
		raise NoBunchException("No next border found!")
	if y_offset is not None:
		y_offset.inc(bunch.elems[-1] + 1)
	return bunch

def do_show_check(msg, param, img):
	cv2.imshow(f"{msg}::{param}", img)
	cv2.waitKey(0)

if __name__ == "__main__":
	from argparse import ArgumentParser
	from pprint import pprint, pp
	import numpy as np
	from pathlib import Path
	from dotenv import dotenv_values
	import tomllib

	parser = ArgumentParser()
	parser.add_argument('--app', default='taimee', help='Application name')
	parser.add_argument('--toml', default='ocr-filter.toml', help='Configuration toml file name')
	parser.add_argument('--file', help='Image file to get parameter')
	args = parser.parse_args()

	cwd = Path(__file__).resolve().parent
	sys.path.append(str(cwd.parent))
	from set_logger import set_logger
	logger = set_logger(__name__)
	import os

	APP_STR = args.app or "taimee"
	if args.file:
		image_fullpath = Path(args.file).resolve()
	elif args.toml:
		try: # if env_file.exists():
			if not args.toml.endswith('.toml'):
				args.toml += '.toml'
			FILTER_TOML = args.toml
			FILTER_TOML_PATH = Path(FILTER_TOML).resolve()
			with FILTER_TOML_PATH.open('rb') as f:
				filter_config = tomllib.load(f)

			image_path_config = filter_config['image-path']
			image_dir = Path(image_path_config['dir']).expanduser() # home dir starts with tilde(~)
			if not image_dir.exists():
				raise ValueError("Error: image_dir not found: %s" % image_dir)
			image_config_filename = image_path_config[APP_STR]['filename']
			filename_path = Path(image_config_filename)
			if '*' in filename_path.stem or '?' in filename_path.stem:
				logger.info("Trying to expand filename with wildcard: %s" % image_config_filename)
				glob_path = Path(image_dir)
				file_list = [f for f in glob_path.glob(image_config_filename) if f.is_file()]
				if len(file_list) == 0:
					raise ValueError("Error: No files found with wildcard: %s" % image_config_filename)
				logger.info("%d file found", len(file_list))
				logger.info("Files: %s", file_list)
				nth = 1
				logger.info("Choosing the %d-th file.", nth)
				image_config_filename = file_list[nth - 1].name
				logger.info("Selected file: %s", image_config_filename)
			image_path = Path(image_dir) / image_config_filename
			if not image_path.exists():
				raise ValueError("Error: image_path not found: %s" % image_path)
			image_fullpath = image_path.resolve()
			filter_area_param_dict = filter_config['ocr-filter']['taimee']
		except KeyError as err:
			raise ValueError("Error: key not found!\n%s" % err)
		except FileNotFoundError as err:
			raise ValueError("Error: file not found!\n%s" % err)
		except tomllib.TOMLDecodeError as err:
			raise ValueError("Error: failed to TOML decode!\n%s" % err)
	else:
		raise ValueError("Image filter module test to extract parameter from an image file. Needs filespec.")
	image = cv2.imread(str(image_fullpath), cv2.IMREAD_GRAYSCALE) #cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
	if image is None:
		raise ValueError("Error: Could not load image: %s" % image_fullpath)
	filter_param_dict: dict[ImageAreaName, ImageFilterParam] = {
		# ImageAreaName.heading:filter_area_param_dict['HeadingAreaParam'],
		ImageAreaName.breaktime:filter_area_param_dict['BreaktimeAreaParam'],
		ImageAreaName.shift:filter_area_param_dict['ShiftAreaParam'],
		ImageAreaName.paystub:filter_area_param_dict['PaystubAreaParam'],
	}
	taimee_filter = TaimeeFilter(image=image, param_dict=filter_param_dict, show_check=True)
	print("[ocr-filter.taimee]")	
	# print(f"{para.__class__.__name__:para.as_toml() for para in taimee_filter.area_param_list}")
	print('\n'.join([param.as_toml() for param in taimee_filter.area_param_list]))
	# --toml ocr-filter
	'''[ocr-filter.taimee]
HeadingAreaParam = [0, 111, 196, -1]
ShiftAreaParam = [219, 267, 345, 373]
BreaktimeAreaParam = [488, 224, 0, 720]
PaystubAreaParam = [714, -1, 0, -1]'''