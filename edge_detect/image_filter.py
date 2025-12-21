from typing import TypedDict
from io import IOBase
from enum import IntEnum, auto
from dataclasses import field
from pathlib import Path
from typing import Iterator, Sequence, NamedTuple, Optional, Sequence
from dataclasses import dataclass
from enum import Enum
import sys

cwd = Path(__file__).resolve().parent
sys.path.append(str(cwd.parent))

import matplotlib.pyplot as plt
from cv2 import UMat
import cv2
from tesseract_ocr import TesseractOCR
from camel_converter import to_snake

# from cvc2.typing import MatLike
# import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from inspect import isclass
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
	y_offset = (KeyUnit.PIXEL, 0)
	heading = (KeyUnit.TEXT, 0)  # heading"
	heading_button = (KeyUnit.TEXT, 1)  # heading"
	work_time = (KeyUnit.HOUR, 0)  # "hours"
	breaktime = (KeyUnit.HOUR, 2)  # "rest_hours"
	shift = (KeyUnit.TEXT, 0)  # "hours_from"
	shift_start = (KeyUnit.TIME, 1)  # "hours_from"
	shift_end = (KeyUnit.TIME, 2)  # "hours_to"
	paystub = (KeyUnit.TEXT, 2)  # "other"
	salary = (KeyUnit.MONEY, 1)  # "salary"


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

	@classmethod
	def min_height(cls):
		return 9

	@classmethod
	def min_width(cls):
		return 9

	def __post_init__(self):
		if self.height is not None:
			if 0 <= self.height < self.min_height():
				raise InvalidValueError(f"height must be larger than {self.min_height()} except negative or None")
		if self.width is not None:
			if 0 <= self.width < self.min_width(): # != -1 and self.width <= 0:
				raise InvalidValueError(f"width must be larger than {self.min_width()} except negative or None")

	@classmethod
	def from_image(cls, image: np.ndarray, offset:int=0) -> "ImageAreaParam":
		height, width = image.shape[:2]
		return cls(height=height, width=width)

	@property
	def param(self)-> Int4:
		return (self.y_offset, self.height or -1, self.x_offset, self.width or -1)


	@classmethod
	def get_class_name(cls):
		return cls.__class__.__name__

	def as_toml(self, **kwargs):
		name = kwargs.get('name', '').strip()
		name_str = f".{name}" if name else ""
		class_name = self.__class__.__name__ # get_class_name()
		class_name_without_area_param = class_name[:-len("AreaParam")]
		class_name_node = to_snake(class_name_without_area_param)
		underscore = class_name_node.rfind("_")
		if underscore > 0:
			class_name_node = class_name_node[underscore+1:]
		return f"{class_name_node}{name_str} = {str(list(self.param))}"

	def as_slice_param(self) -> Sequence[Int4]:
		return ((self.y_offset, (self.y_offset + self.height) if self.height and self.height > 0 else -1, self.x_offset, (self.x_offset + self.width) if self.width and self.width > 0 else -1),)

	def crop_image(self, image: np.ndarray, y_margin: int = 0) -> Iterator[np.ndarray]:
		for param in self.as_slice_param():
			y_start = y_margin + param[0]
			y_stop = y_margin + param[1] if param[1] > 0 else param[1]
			# h = y_margin - param[1] if param[1] < 0 else y_margin + param[0] + param[1]
			yield image[y_start:y_stop, param[2]:param[3]]

	def to_toml(self, fp: IOBase, **kwargs):
		fp.write(self.as_toml(**kwargs) + '\n')



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

# camel-converter
@dataclass
class HeadingAreaParam(ImageAreaParam):
	'''
	Necessary named parameters: height, xpos'''

	@classmethod
	def get_class_name(cls):
		return cls.__class__.__name__

	@classmethod
	def get_label_text_start(cls) -> str:
		return "LABEL TEXT START"


# from taimee_filter import TaimeeHeadingAreaParam

def get_center_run_length(line: Sequence[int]) -> int | None:
	kg_list = [(k, len(list(g))) for n, (k, g) in enumerate(groupby(line)) if n < 3]
	if not kg_list:
		raise ValueError("Empty list")
	if len(kg_list) == 1 and kg_list[0][0] == 255 and kg_list[0][1] == len(line):
		return -1 # white
	if len(kg_list) == 2:
		return
	if sum([g for k, g in kg_list]) < len(line):
		return
	return kg_list[1][1]


@dataclass
class ShiftAreaParam(ImageAreaParam):
	'''Needs to initialize using named parameters::
	start_width: as xpos
	end_xpos: as width
	while start_xpos is 0 and end_width is -1
	'''
	@classmethod
	def get_class_name(cls):
		return cls.__class__.__name__


	@classmethod
	def check_image(cls, image: np.ndarray, image_check:bool=False)-> tuple[int, int]:
		'''returns (left_end, right_start)'''
		# check if avatar circle at the left side of the area between the borders(1st and 2nd)
		## scan vertical lines to find the horizontal range		 of the shape (expetcing as a black filled rectangle of left side flat)
		x_center = image.shape[1] // 2
		def trim_left() -> int | None:
			for x in range(x_center, 0, -1):
				if np.any(image[:, x] == 0):
					break
			if not (runlen:= get_center_run_length(image[:, x].tolist())):
				return
			for x0 in range(x, 0, -1):
				# runlen = None
				line = image[:, x0].tolist()
				runlen = get_center_run_length(line)
				if runlen is not None and runlen < 0:
					return x0
		# left = trim_left()
		def both_side_diff() -> int:
			last_zeros = 0
			stable = False
			b = 0
			for b in range(1, x_center):
				non_zeros = np.count_nonzero(image[:, x_center - b:x_center + b])
				zeros = image[:, x_center - b:x_center + b].size - non_zeros
				if zeros == 0:
					continue
				if zeros == last_zeros:
					stable = True
					break 
				last_zeros = zeros
			return b if (b > 0 and stable) else -1
		b = both_side_diff()
		if b < 0:
			raise ValueError("Center shape not found!")
		image[:, x_center - b] = 0
		return x_center - b, x_center + b
		'''all_white = False

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
			cv2.imshow("Shift area center", image[:, x0:x])
			cv2.waitKey(0)
		return x0, x'''

	@classmethod
	def from_image(cls, image: np.ndarray, offset_range: range, image_check:bool=False) -> "ShiftAreaParam":
		left, right = cls.check_image(image=image[offset_range.start:offset_range.stop, :], image_check=image_check)
		return cls(y_offset=offset_range.start, height=offset_range.stop - offset_range.start, x_offset=left, width=right)


	@property
	def start_width(self)-> int: # start-from time is until here
		return self.x_offset

	@property
	def end_offset(self)-> int: # end-by time is from here
		return self.width

	def as_slice_param(self) -> Sequence[Int4]:
		return ((self.y_offset, self.y_offset + self.height, 0, self.x_offset),(self.y_offset, self.y_offset + self.height, self.width, -1))


@dataclass
class BreaktimeAreaParam(ImageAreaParam):
	@classmethod
	def get_class_name(cls):
		return cls.__class__.__name__


@dataclass
class PaystubAreaParam(ImageAreaParam):
	@classmethod
	def get_class_name(cls):
		return cls.__class__.__name__

@dataclass
class SalaryAreaParam(ImageAreaParam):
	@classmethod
	def get_class_name(cls):
		return cls.__class__.__name__



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
				fp.write(f"{key.name} = ")
				fp.write(f"{area.as_dict()}\n") if as_dict else fp.write(f"{list(area.param)}\n")

	area_key_list = [ImageDictKey.y_offset, ImageDictKey.heading, ImageDictKey.shift_start, ImageDictKey.shift_end, ImageDictKey.breaktime, ImageDictKey.paystub, ImageDictKey.salary]

	areas = {
		ImageDictKey.y_offset: OffsetArea,
		ImageDictKey.heading: HeadingAreaParam,
		ImageDictKey.shift: ShiftAreaParam,
		ImageDictKey.breaktime: BreaktimeAreaParam,
		ImageDictKey.paystub: PaystubAreaParam,
		ImageDictKey.salary: SalaryAreaParam,
	}
	heading: HeadingAreaParam # midashi
	shift: ShiftAreaParam # syuugyou jikan
	break_time: BreaktimeAreaParam # kyuukei jikan
	paystub: PaystubAreaParam # meisai
	salary: SalaryAreaParam # kyuuyo
	y_offset: int = 0

class ImageAreaParamName(Enum):
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
	return ShiftAreaParam(y_offset=0, height=image.shape[0], start_width=left_area_width, end_offset=right_area_xpos) if return_as_cuts else (image[:, :left_area_width], image[:, right_area_xpos:])



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
	if __debug__:
		cv2.imshow(f"{msg}::{param}", img)
		cv2.waitKey(0)

from argparse import ArgumentParser
# from dotenv import dotenv_values
import tomllib
from fnmatch import fnmatch

class APP_NAME(Enum):
	TAIMEE = '_jp.co.taimee'
	MERCARI = '_jp.mercari.work.android'

	def __str__(self):
		return self.name.lower()

class AppNameToEnum(TypedDict):
	key: str
	value: APP_NAME

app_name_to_enum: AppNameToEnum 
app_name_to_enum = {n.name.lower(): n for n in APP_NAME}  # type: ignore

def main():
	from taimee_filter import TaimeeFilter
	OCR_FILTER = "ocr-filter"
	parser = ArgumentParser()
	parser.add_argument('files', nargs='*', help='Image files to commit OCR or to get parameters. Specify like: *.png')
	parser.add_argument('--app', choices=[n.name.lower() for n in APP_NAME], type=str, help=f'Application name of the screenshot to execute OCR:(specify in TOML filename =: {[f"*{n.value}.png" for n in APP_NAME]})') # 
	parser.add_argument('--toml', help=f'Configuration toml file name like {OCR_FILTER}')
	parser.add_argument('--save', help='Output path to save OCR text of the image file as TOML format into the image file name extention as ".ocr-<app_name>.toml"')
	parser.add_argument('--dir', help='Image dir of files: ./')
	parser.add_argument('--nth', type=int, default=1, help='Rank(default: 1) of files descending sorted(the latest, the first) by modified date as wildcard(*, ?)')
	parser.add_argument('--glob-max', type=int, default=60, help='Pick up file max as pattern found in TOML')
	parser.add_argument('--show', action='store_true', help='Show images to check')
	parser.add_argument('--make', help=f'make config. from image(i.e. this arg. makes not to load a config file like "{OCR_FILTER}.toml")')
	parser.add_argument('--no-ocr', action='store_true', default=False, help='Do not execute OCR')
	parser.add_argument('--ocr-conf', type=int, default=55, help='Confidence threshold for OCR')
	parser.add_argument('--psm', type=int, default=6, help='PSM value for Tesseract')
	args = parser.parse_args()
	# if not args.files: parser.print_help() sys.exit(1)
	filter_area_param_dict = {}
	# image_config_filename = (args.file) #.resolve()Path
	filter_config_is_loaded = False
	def get_filter_config(filter_config = {}):
		nonlocal filter_config_is_loaded
		if args.toml and not filter_config_is_loaded:
			try: # if env_file.exists():
				if not args.toml.endswith('.toml'):
					args.toml += '.toml'
				filter_toml_path = Path(args.toml) # .resolve()
				with filter_toml_path.open('rb') as f:
					filter_config |= tomllib.load(f)
				logger.info("Loaded filter config in [%s]", filter_toml_path, filter_config)
			except FileNotFoundError:
				logger.warning("FileNotFoundError in TOML")
			except tomllib.TOMLDecodeError:
				logger.warning("Not a valid TOML file")
			filter_config_is_loaded = True
		return filter_config

	file_list_loaded = False
	def get_args_files(file_list=[]):
		if len(args.files) == 1:
			return args.files[0]
		else:
			nonlocal file_list_loaded
			if not file_list_loaded:
				_file_list = [(Path(f), Path(f).stat().st_mtime) for f in args.files]
				file_list += [m[0] for m in sorted(_file_list, key=lambda f: f[1], reverse=True)]
				file_list_loaded = True
				logger.info("Loaded file_list of %d files: %s", len(file_list), file_list)
			return file_list

	class NoAppNameException(Exception):
		"""No application name specified"""
		pass

	if args.app is not None:
		try:
			app_name = app_name_to_enum[args.app]
		except KeyError:
			sys.exit(f"No such app name:{args.app}")
	else:
		try: # try to extract app name from file name
			_file = get_args_files()[args.nth - 1]
			app_name = None
			for nm in APP_NAME:
				if _file.stem.endswith(nm.value):
					app_name = nm
					break
			if not app_name:
				raise NoAppNameException()	
			logger.info("Application name is set as '%s' from file name:%s", app_name.name, _file.name)
		except (IndexError, NoAppNameException):
			sys.exit("Needs application name spec. by '--app' option or file name(ending with {}).".format([nm.value for nm in APP_NAME]))
	try:
		image_path_dir = Path(args.dir or get_filter_config()['image-path']['dir'])
		logger.info("Image directory is set as %s by %s", image_path_dir, 'args.dir' if args.dir else 'config.image_path.dir')
		if str(image_path_dir)[0] == '~':
			try:
				image_path_dir = image_path_dir.expanduser()
			except RuntimeError:
				sys.exit("image_dir.expanduser() failed!")
			logger.info("Image directory is expanded user by args as %s", image_path_dir)
	except (TypeError, KeyError):
		image_path_dir = None

	if image_path_dir and not image_path_dir.exists():
		sys.exit("Image dir. does not exist: %s" % image_path_dir)

	if not get_args_files() and args.toml:
		try:
			image_file_pattern = get_filter_config()['image-path'][str(app_name)]['filename']
			logger.info("Image filename pattern is set by %s as: %s", args.toml, image_file_pattern)
			is_wildcard = False
			for c in "*?[]":
				if c in image_file_pattern:
					is_wildcard = True
					break
			if is_wildcard:
				_file_list = []
				# search_path = (image_path_dir) / image_file_pattern if image_path_dir else Path(image_file_pattern)
				for n, match_f in enumerate(image_path_dir.glob(image_file_pattern)):
					_file_list.append(match_f)
					if n > args.glob_max:
						logger.warning("Exceeded glob_max: %d", args.glob_max)
						break
				if len(_file_list) == 0:
					sys.exit("Error: No files found %s: %s" % ('with wildcard' if is_wildcard else 'without wildcard', image_file_pattern))
				else:
					logger.info("%d files are listed.", len(_file_list))
				file_list = sorted(_file_list, key=lambda f: f.stat().st_mtime, reverse=True)
		except KeyError:
			sys.exit("No files found in toml file as [image-path.%s]\nfilename='*_jp.co.taimee.png'" % str(app_name))
		'''else:
			logger.info("%d file%s found from %s", len(file_list), 's' if len(file_list) > 1 else '', image_file_pattern)
			from os.path import getmtime
			nth = args.nth # if args.nth else 1
			logger.info("Choosing the %d-%s file from the latest in %d files.", nth, "th" if nth > 3 else ("st", "nd", "rd")[nth - 1], len(file_list))
			file_list.sort(key=getmtime, reverse=True)'''
	else:
		file_list = get_args_files() # [Path(f) for f in args.files]
		# raise ValueError("Error: Image file name is not specified with --file option or not found in toml file as [image-path.%s]\nfilename='*_jp.co.taimee.png'" % app_name)
	# filename_path = Path(image_file_pattern)
#glob_path.glob(str(filename_path)) if f.is_file()]
		# logger.info("Files: %s", file_list)
		# sort by modified date descendingf
	try:
		image_file = file_list[args.nth - 1]
	except IndexError:
		# image_file = file_list[0]
		sys.exit(f"Index out of range for file_list by {args.nth=}")
	logger.info("Selected file: %s", image_file.name)
	# else: image_file = filename_path
	if not image_file.exists():
		sys.exit("Error: image_file not found: %s" % image_file)
	# image_fullpath = image_path.resolve()
	try:
		filter_area_param_dict = {} if args.make else get_filter_config()[OCR_FILTER][str(app_name)]
	except KeyError:
		filter_area_param_dict = {}
		if not args.make:
			logger.warning("KeyError: '%s' not found in get_filter_config()", app_name)

	image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE) #cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
	if image is None:
		raise ValueError("Error: Could not load image: %s" % image_file)
	filter_param_dict: dict[ImageAreaParamName, dict[str, str|float|None]] = {}
	if not args.make:
		for key in ImageAreaParamName:
			try:
				filter_param_dict[key] = filter_area_param_dict[key.name]
			except KeyError:
				filter_param_dict[key] = {}
		'''ImageAreaParamName.heading:filter_area_param_dict.get('heading'),
		ImageAreaParamName.breaktime:filter_area_param_dict.get('breaktime'),
		ImageAreaParamName.shift:filter_area_param_dict.get('shift'),
		ImageAreaParamName.paystub:filter_area_param_dict.get('paystub'),'''

	# print(f"{para.__class__.__name__:para.as_toml() for para in taimee_filter.area_param_list}")
	if not args.no_ocr:
		match app_name:
			case APP_NAME.TAIMEE:
				app_filter = TaimeeFilter(image=image, param_dict=filter_param_dict, show_check=args.show)
			case APP_NAME.MERCARI:
				sys.exit("Error: this app_name is not yet implemented: %s" % app_name)
			case _:
				sys.exit("Unknown app_name : %s" % app_name)
		from tesseract_ocr import TesseractOCR, Output
		from tomli_w import dumps
		# from tomlkit import TOMLDocument, table, comment, nl, dump
		ocr = TesseractOCR()
		doc_dict = {}
		# doc = TOMLDocument() doc.add(comment("ocr-result")) doc.add(nl()) doc.add(comment(image_file.name)) doc.add(nl())
		print(f"# {image_file.name=}")
		for k, area_param in app_filter.area_param_dict.items():
			area_name = f"ocr-{k.name}"
			# area_tbl = table() area_tbl.add(comment(area_name)) area_tbl.add(nl())
			area_dict = {}
			pg_list = []
			print(f"[{area_name}]")
			for pg, ocr_area in enumerate(area_param.crop_image(app_filter.image, app_filter.y_margin)):
				pg_str = f"p{pg+1}"
				ocr_result = ocr.exec(ocr_area, output_type=Output.DATAFRAME, psm=args.psm)
				max_line = max(ocr_result['line_num'])
				def textline(n, conf=args.ocr_conf):
					return ocr_result[(ocr_result['line_num'] == n) & (ocr_result['conf'] > conf)]# ['text'] # return ''.join(
				df_list = []
				for r in range(1, max_line + 1):
					line = textline(r) 
					df_list.append(line) # '\n'.join(line))
				ocr_text = [' '.join(df['text']) for df in df_list]
				print(f"{pg_str}={ocr_text}")
				pg_list.append(ocr_text)
				# area_tbl.add(comment(pg_str)) area_tbl.add(nl()) area_tbl.add(comment(ocr_text)) area_tbl.add(nl())
			if len(pg_list) > 1:
				for p, pg in enumerate(pg_list):
					pg_str = f"p{p+1}"
					area_dict[pg_str] = '\n'.join(pg) 
				doc_dict[area_name] = area_dict
			else:
				doc_dict[area_name] = '\n'.join(pg_list[0])
			# doc.add(area_tbl)
		if args.save:
			save_path = Path(args.save) / (image_file.stem + '.ocr-' + app_name.name.lower() + '.toml')
			if save_path.exists():
				yn = input(f"\nThe file path to save the image file area configuration:'{save_path}'\n already exists. Overwrite?(Enter 'Yes' or 'Affirmative' if you want to overwrite)").lower()
				if yn != 'yes' and yn != 'affirmative':
					sys.exit("Exit since the user not accept overwrite of: %s" % save_path)
			toml_text = dumps(doc_dict, multiline_strings=True)
			with save_path.open('w') as wf:
				wf.write(toml_text)
			logger.info("Saved toml file into: %s\n%s", save_path, toml_text)

#
	if args.make:
		make_path = Path(args.make + '.toml') if not args.make.endswith('.toml') else Path(args.make)
		if make_path.exists():
			try:
				yn = input(f"\nThe file path to save the image file area configuration:'{make_path}'\n already exists. Overwrite?(Enter 'Yes' or 'Affirmative' if you want to overwrite)").lower()
			except (IndexError, EOFError, KeyboardInterrupt):
				yn = 'n'
			if yn != 'yes' and yn != 'affirmative':
				sys.exit("Exit since the user not accept overwrite of: %s" % make_path)
		from io import StringIO
		sio = StringIO()
		with make_path.open('w') as wf: # dump(doc, wf)
			label = f"[ocr-filter.{str(app_name)}]"
			print(label, file=wf)
			print(f"[ocr-filter.{label}]", file=wf)
			for key, param in app_filter.area_param_dict.items():
				param.to_toml(wf)
				sio.write(param.as_toml() + '\n')
		sio.seek(0)
		logger.info("Image area parameters are saved into %s\nas: %s", make_path, sio.read())
		# print("[ocr-filter.taimee]")	
		# print(sio.read())
	# --toml ocr-filter
	'''[ocr-filter.taimee]
HeadingAreaParam = [0, 111, 196, -1]
ShiftAreaParam = [219, 267, 345, 373]
BreaktimeAreaParam = [488, 224, 0, 720]
PaystubAreaParam = [714, -1, 0, -1]'''

if __name__ == "__main__":
	main()