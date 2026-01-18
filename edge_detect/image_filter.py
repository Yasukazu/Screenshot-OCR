from configparser import ConfigParser
from itertools import groupby
from typing import Type, TypedDict
from io import IOBase
from pathlib import Path
from datetime import date
from typing import Iterator, Sequence, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from datetime import date as Date
import sys
import atexit
from tomlkit import TOMLDocument
from dotenv import load_dotenv

from cv2 import UMat
import cv2
from returns.result import safe #, attempt, Result, Failure, Success
from returns.pipeline import is_successful
# from returns.primitives.exceptions import UnwrapFailedError
# from returns.io import IO
from camel_converter import to_snake

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from inspect import isclass
from peewee import OperationalError
from fancy_dataclass import TOMLDataclass

cwd = Path(__file__).resolve().parent
sys.path.insert(0, str(cwd.parent))
from set_logger import set_logger
logger = set_logger(__name__)

class APP_NAME(StrEnum):
	''' name of app: value is stem end '''
	TAIMEE = auto() # '_jp.co.taimee'
	MERCARI = auto() # '_jp.mercari.work.android'

	def __str__(self):
		return self.name.lower()

class AppNameToEnum(TypedDict):
	key: str
	value: APP_NAME
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

class GetAreaParamException(Exception):
	pass

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

	@classmethod
	def from_str(cls, str: str) -> "ImageAreaParam":
		return cls.from_param([int(i) for i in str.strip('[]').split(',')])

	@classmethod
	def from_param(cls, param: Sequence[int]) -> "ImageAreaParam":
		''' new from a sequence of int '''
		match len(param):
			case num if num in range(1, 4):
				return cls(*(list(param) + [-1, 0, -1][:num - 4]))
			case _:
				return cls(*param)

	def __post_init__(self):
		if self.height is not None:
			if 0 <= self.height < self.min_height():
				raise InvalidValueError(f"height must be larger than {self.min_height()} except negative or None")
		if self.width is not None:
			if 0 <= self.width < self.min_width(): # != -1 and self.width <= 0:
				raise InvalidValueError(f"width must be larger than {self.min_width()} except negative or None")

	@classmethod
	def from_image(cls, image: np.ndarray, offset:int=0, offset_range: range=range(0), image_check:bool=False) -> "ImageAreaParam":
		# height, width = image.shape[:2]
		# return cls(height=height, width=width)
		# manual(GUI)
		from mouse_event import get_area, QuitKeyException

		try:
			TLpos, BRpos = get_area(f"{cls.__name__}", image)
		except QuitKeyException as e:
			raise GetAreaParamException() from e
		return cls(y_offset=TLpos[1], height = BRpos[1] - TLpos[1], x_offset=BRpos[0])#offset_range.start, height=offset_range.stop - offset_range.start)

	@property
	def param(self)-> list[int]:
		return [self.y_offset, self.height or -1, self.x_offset, self.width or -1]


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
		if image.size == 0:
			raise ValueError("Image size is 0")
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
	''' adding to super, x_offset2 for shift end '''
	x_offset2: int = 0

	@property
	def param(self)-> list[int]:
		return [*super().param, self.x_offset2]

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


	@classmethod
	def from_image(cls, image: np.ndarray, offset_range: range, image_check:bool=False) -> "ShiftAreaParam":
		left, right = cls.check_image(image=image[offset_range.start:offset_range.stop, :], image_check=image_check)
		return cls(y_offset=offset_range.start, height=offset_range.stop - offset_range.start, x_offset=0, width=left, x_offset2=right)


	@property
	def start_width(self)-> int: # start-from time is until here
		return self.x_offset

	@property
	def end_offset(self)-> int: # end-by time is from here
		return self.width

	def as_slice_param(self) -> Sequence[Int4]:
		return (self.y_offset, self.y_offset + self.height, self.x_offset, self.x_offset + self.width), (self.y_offset, self.y_offset + self.height, self.x_offset2, -1)


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

class ImageAreaParamName(StrEnum):
	HEADING = auto() # HeadingAreaParam
	SHIFT = auto() # ShiftAreaParam
	BREAKTIME = auto() # BreaktimeAreaParam
	PAYSTUB = auto() # PaystubAreaParam
	SALARY = auto() # SalaryAreaParam
	#@classmethod
	def to_param_class(self) -> Type[ImageAreaParam]:#, name: str):
		match self:
			case self.HEADING:
				return HeadingAreaParam
			case self.SHIFT:
				return ShiftAreaParam
			case self.BREAKTIME:
				return BreaktimeAreaParam
			case self.PAYSTUB:
				return PaystubAreaParam
			case self.SALARY:
				return SalaryAreaParam
			case _:
				raise ValueError(f"Invalid enum value: {self}")

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
		# print("Press a key to continue...")
		plt.waitforbuttonpress() # stand by here
		# print("Key pressed! Continuing...")
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

sys.path.insert(0, str(Path(__file__).parent))
from near_bunch import NearBunch, NearBunchException, NoBunchException

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


app_name_to_enum: AppNameToEnum 
app_name_to_enum = {n.name.lower(): n for n in APP_NAME}  # type: ignore

class MainError(ValueError):
	''' base error of main '''
	def __init__(self, msg: str):
		self.msg = msg

class ConfigError(MainError):
	''' Bad config. '''
	def __init__(self, msg: str):
		self.msg = msg

class MakeError(MainError):
	''' Error during make process. '''
	def __init__(self, msg: str):
		self.msg = msg

class MainException(Exception):
	''' base exception of main '''
	def __init__(self, msg: str):
		self.msg = msg

class KeyException(MainException):
	''' Key probrem '''
	def __init__(self, msg: str, key: str):
		self.msg = msg
		self.key = key

class SectionKeyException(KeyException):
	''' Section Key probrem '''
	def __init__(self, msg: str, key: str, section: str):
		super().__init__(msg, key)
		self.section = section

class ConfigKeyException(KeyException):
	''' Config Section probrem '''
	def __init__(self, msg: str, key: str, config: ConfigParser):
		super().__init__(msg, key)
		self.key = key
		self.config = config

from os.path import abspath, dirname
from pydantic_settings import BaseSettings, SettingsConfigDict, CliApp

class Settings(BaseSettings):
	env_prefix:str = "image_filter_"
	env_file:str = "image-filter.env"
	env_file_encoding:str = "utf-8"
	use_enum_values:bool = True
	image_ext: str = ".png"
	image_ext_set: set[str] = {".jpg", ".png"}
	app_name_to_stem_end:dict[str, str] = {}#'taimee': '_jp.co.taimee', 'mercari': '_jp.mercari.work.android'}
	model_config = SettingsConfigDict(env_prefix=env_prefix, env_file=env_file, env_file_encoding=env_file_encoding, use_enum_values=use_enum_values)

	def cli_cmd(self) -> None:
		return self.image_ext

from configparser import ConfigParser, NoSectionError, SectionProxy
from configargparse import ArgParser, CompositeConfigParser, TomlConfigParser, IniConfigParser, ConfigparserConfigFileParser
from os.path import join as os_path_join
from typing import Any
from dotenv import load_dotenv

class ConfigFileExt(StrEnum):	
	TOML = auto()
	INI = auto()
	CFG = auto()
	CONF = auto()

class NoAppNameError(ConfigError):
	"""No application name specified"""
	pass



IMAGE_AREA_PARAM_STR = "image_area_param"
def main(
	config_dir = abspath(dirname(__file__)), config_file_stem = "image-filter", config_file_ext_enum = ConfigFileExt, image_area_param_file_stem = IMAGE_AREA_PARAM_STR.replace('_', '-'),
	env_file = "image-filter.env",
):
	load_dotenv(f"{config_dir}/{env_file}")
	from taimee_filter import TaimeeFilter

	APP_NAME_TO_FILTER_CLASS = {APP_NAME.TAIMEE: TaimeeFilter}
	OCR_FILTER = "ocr-filter"
	config_sections = [
		"app_stem_end",
		"common",
	]  # [ IMAGE_AREA_PARAM_STR + '.' + app.value for app in APP_NAME]
	default_config_paths = [
		Path(config_dir) / f"{stem}.{ext.lower()}"
		for ext in config_file_ext_enum
		for stem in [config_file_stem]
	]  # image_area_param_file_stem]]# if f.exists()]
	default_config_files = [p for p in default_config_paths if p.exists()]
	parser = ArgParser(
		default_config_files=default_config_files,
		config_file_parser_class=CompositeConfigParser(
			[
				TomlConfigParser(config_sections),
				IniConfigParser(config_sections, split_ml_text_to_list=True),
			]
		),
	)
	parser.add_argument(
		"--image_ext", nargs="+", default=[".png"], env_var="IMAGE_FILTER_IMAGE_EXT"
	)
	parser.add_argument(
		"--image_dir",
		default="~/Documents/screenshots",
		env_var="IMAGE_FILTER_IMAGE_DIR",
		type=Path,
	)
	parser.add_argument(
		"--shot_month",
		action="append",
		type=int,
		env_var="IMAGE_FILTER_SHOT_MONTH",
		help='Choose Screenshot file by its month (MM part of [YYYY-MM-DD or YYYYMMDD]) included in filename stem. {Jan. is 01, Dec. is 12}(specified in a list like "[1,2,..]" )',
	)
	parser.add_argument(
		"files",
		nargs="*",
		help="Image file fullpaths to commit OCR or to get parameters.",
	)
	parser.add_argument(
		"--app_stem_end",
		env_var="IMAGE_FILTER_APP_STEM_END",
		default="taimee:_jp.co.taimee mercari:_jp.mercari.work.android",
		help='Screenshot image file name pattern of the screenshot to execute OCR:(specified in format as "<app_name1>:<stem_end1>,<stem_end2> ..." )',
	)
	parser.add_argument(
		"--image_area_param_section_stem",
		env_var="IMAGE_AREA_PARAM_SECTION_STEM ",
		default="image_area_param",
	)
	parser.add_argument(
		"--app_border_ratio",
		env_var="IMAGE_FILTER_APP_BORDER_RATIO",
		default="taimee:2.2,3.2",
		nargs="*",
		help='Screenshot image file horizontal border ratio list of the app to execute OCR:(specified in format as "<app_name1>:<ratio1>,<ratio2> ..." )',
	)
	parser.add_argument(
		"--app_suffix",
		action="store_true",
		default=True,
		help='Screenshot image file name has suffix(sub extention) of the same as app name i.e. "<stem>.<suffix>.<ext>" (default: True)',
	)

	# parser.add_argument('--filename_pattern', action='append', default=['*{app_stem_end}{image_ext}'], help='Image files to commit OCR or to get parameters. Can be specified multiple times. Default is: *{app_stem_end}{image_ext}')
	parser.add_argument(
		"--app",
		choices=[n.name.lower() for n in APP_NAME],
		env_var="IMAGE_FILTER_APP_NAME",
		help=f"Application name of the screenshot to execute OCR: choices={', '.join(n.name.lower() for n in APP_NAME)}",
	)  #
	# parser.add_argument('--toml', help=f'Configuration toml file name like {OCR_FILTER}')
	parser.add_argument(
		"--save",
		help='Output path to save OCR text of the image file as TOML format into the image file name extention as ".ocr-<app_name>.toml"',
	)
	# parser.add_argument('--dir', help='Image dir of files: ./')
	parser.add_argument(
		"--nth",
		type=int,
		default=1,
		help="Rank(default: 1) of files descending sorted(the latest, the first) by modified date as wildcard(*, ?)",
	)
	parser.add_argument(
		"--glob-max",
		type=int,
		default=60,
		help="Pick up file max as pattern found in TOML",
	)
	parser.add_argument("--show", action="store_true", help="Show images to check")
	parser.add_argument(
		"--make",
		help='make a image area param config file from image in TOML format(i.e. this arg. makes not to use param configs in any config file;  specify image_area_param values like "--image_area_param heading:0,106,196,-1"',
	)
	parser.add_argument(
		"--no-ocr", action="store_true", default=False, help="Do not execute OCR"
	)
	parser.add_argument(
		"--ocr-conf", type=int, default=55, help="Confidence threshold for OCR"
	)
	parser.add_argument("--psm", type=int, default=6, help="PSM value for Tesseract")
	parser.add_argument(
		"--area_param_dir",
		help="Screenshot image area parameter config file directory",
		type=Path,
		env_var="IMAGE_FILTER_AREA_PARAM_DIR",
	)
	parser.add_argument(
		"--area_param_file",
		help='Screenshot image area parameter config file: format as INI or TOML(".ini" or ".toml" extention respectively): in [image_area_param.<app>] section, items as "<area_name>=[<p1>,<p2>,<p3>,<p4>]" (e.g. "heading=[0,106,196,-1]")',
		type=Path,
		env_var="IMAGE_FILTER_AREA_PARAM_FILE",
		default="image-area-param.ini",
	)
	parser.add_argument(
		"--ocr_filter_sqlite_db_name",
		env_var="OCR_FILTER_SQLITE_DB_NAME",
		default="ocr-filter.db",
		help="SQLite DB file is created under `image_dir`/{yyyy} directory(yyyy is like 2025)",
	)
	parser.add_argument(
		"--data_year",
		env_var="OCR_FILTER_DATA_YEAR",
		type=int,
		default=0,
		help="Year for DB data (like 2025)",
	)
	parser.add_argument(
		"--show_ocr_area",
		action="store_true",
		default=False,
		help="Show every area before commit OCR",
	)
	parser.add_argument(
		"--exclude_area_param_set",
		help=f"Exclude a set of image area parameter names : { {f'{n}' for n in list(ImageAreaParamName)} }",
		nargs='*',
		env_var="IMAGE_FILTER_AREA_PARAM_NAME_EXCLUDE_SET",
	)
	args = parser.parse_args()

	if args.exclude_area_param_set:
		try:
			args.exclude_area_param_set = set(ImageAreaParamName(p) for p in args.exclude_area_param_set)
		except ValueError:
			logger.error("Invalid exclude_area_param_set: %s", args.exclude_area_param_set)
			raise ConfigError("Invalid exclude_area_param_set: %s" % args.exclude_area_param_set)
	else:
		args.exclude_area_param_set = set()

	def get_data_year(month: int = 0):
		if args.data_year == 0:
			if not month:
				raise ConfigError("(data_year and month) are not specified")
			cur_year = Date.today().year
			cur_month = Date.today().month
			"""month_array = list(range(1, 13))
			cur_index = month_array.index(cur_month)
			arg_index = month_array.index(month)"""
			if month <= cur_month:
				return cur_year
			else:
				return cur_year - 1
		return args.data_year

	if args.area_param_dir:
		try:
			args.area_param_dir = Path(args.area_param_dir).expanduser()
		except TypeError:
			logger.error("Invalid area_param_dir: %s", args.area_param_dir)
			raise ConfigError(f"Invalid area_param_dir: {args.area_param_dir}")
		except RuntimeError:
			logger.error("Failed to expand area_param_dir: %s", args.area_param_dir)
			raise ConfigError(f"Failed to expand area_param_dir: {args.area_param_dir}")
	if args.area_param_file:
		if args.area_param_file.name[0] == "~":
			try:
				args.area_param_file = Path(args.area_param_file[0]).expanduser()
			except RuntimeError:
				logger.error("Failed to expand area param file.")
				raise ConfigError("Failed to expand area param file.")
		else:
			if args.area_param_dir:
				args.area_param_file = args.area_param_dir / args.area_param_file
			else:
				args.area_param_file = Path(args.area_param_file)
			logger.info("area param file is set: %s", args.area_param_file)
		if not args.area_param_file.exists():
			logger.error("area param file does not exist.")
			raise ConfigError("area param file does not exist.")

	is_app_to_stem_end_set = False

	def get_app_to_stem_end_dict(
		app_to_stem_end_dict: dict[APP_NAME, set[str]] = {},
		stem_end_to_app_dict: dict[str, APP_NAME] = {},
	) -> tuple[dict[APP_NAME, set[str]], dict[str, APP_NAME]]:
		nonlocal is_app_to_stem_end_set
		if is_app_to_stem_end_set:
			return app_to_stem_end_dict, stem_end_to_app_dict
		try:
			for it in args.app_stem_end.split(";"):
				try:
					name, val = it.split(":")
				except ValueError:
					logger.error("Invalid app_stem_end: %s", it)
					raise ConfigError(f"Invalid app_stem_end: {it}")
				vals = val.split(",")
				app_to_stem_end_dict[APP_NAME(name)] = set(vals)
				for val in vals:
					stem_end_to_app_dict[val] = APP_NAME(name)
		except AttributeError:
			if args.app_suffix:
				for app in APP_NAME:
					app_to_stem_end_dict[app] = {app.value}
					stem_end_to_app_dict[app.value] = app
			else:
				raise ConfigError(
					"app_stem_end is not specified and app_suffix is not set"
				)
		is_app_to_stem_end_set = True
		return app_to_stem_end_dict, stem_end_to_app_dict

	def app_to_stem_end_set(app: APP_NAME) -> set[str]:
		return get_app_to_stem_end_dict()[0][app]

	def stem_to_app(stem: str) -> APP_NAME:
		for k, v in get_app_to_stem_end_dict()[1].items():
			if stem.endswith(k) or k in stem.split("."):
				return v
		raise ConfigError(f"Invalid stem: {stem}")

	if not args.app:
		if not args.files:
			from prompt_toolkit.shortcuts import choice

			args.app = APP_NAME(
				choice(
					message="Choose an application:",
					options=[
						(
							n.name.lower(),
							{"taimee": "Taimee Job", "mercari": "Mercari Work"}[
								n.value
							],
						)
						for n in APP_NAME
					],
				)
			)
			logger.info("args.app is chosen by user : %s", args.app)
		else:
			args.app = stem_to_app(Path(args.files[args.nth - 1]).stem)
			logger.info("args.app is chosen by file suffix: %s", args.app)
	else:
		try:
			args.app = APP_NAME(args.app)
		except ValueError:
			raise ConfigError(
				f"Invalid application name: {args.app}."
			)

	def select_area_param(
		area_param_name: ImageAreaParamName,
		image: np.ndarray
	) -> ImageAreaParam:

		if image is None or image.size == 0:
			logger.error("Image is None or size 0")
			raise ValueError("Image is None or size 0")
		logger.info("Try to get area params [%s] from image: %s", area_param_name, image.shape)
		from mouse_event import get_area, QuitKeyException
		try:
			TL, BR = get_area(area_param_name, image)
		except QuitKeyException:
			logger.warning(
				"Failed to get area from image for %s", area_param_name
			)
		else:
			return ImageAreaParam(
				TL[1], BR[1] - TL[1], TL[0], BR[0] - TL[0]
			)

	if not args.files:
		try:
			suffix_list = [APP_NAME(args.app).value]

		except ValueError:
			suffix_list = []
		from path_chooser import ImageFileFeeder

		file_feeder = ImageFileFeeder(suffix_list=suffix_list)
		dir_file_date_list = []
		try:
			dir_file_date_list = list(
				file_feeder.feed(
					Path(args.image_dir),
					month_list=args.shot_month or [m + 1 for m in range(12)],
				)
			)
		except TypeError:
			logger.error("Invalid image_dir: %s", args.image_dir)
			raise ConfigError("Invalid image_dir: %s" % args.image_dir)
		else:
			logger.info(
				"%s files are chosen by feeder with date: %s",
				len(dir_file_date_list),
				[
					d.isoformat()
					for d in set([d for _, fd in dir_file_date_list for f, d in fd])
				],
			)
		args.files = sorted(
			set([(Path(dr) / f) for dr, fd in dir_file_date_list for f, _ in fd]),
			key=lambda f: ImageFileFeeder.pick_date(f.stem) or date.min,
			reverse=True,
		)
	if not args.files:
		logger.info("No files are chosen by feeder")
		raise ConfigError("No files are chosen by feeder")

	_image_area_params: SectionProxy | None = None
	# IMAGE_AREA_PARAM_SECTION_STEM: str = "image_area_param"

	@safe
	def get_image_area_params_section(
		app=args.app,
		section_stem=args.image_area_param_section_stem,
		area_param_file=args.area_param_file,
	) -> SectionProxy:
		"""Get image area parameters' section of ConfigParser"""
		nonlocal _image_area_params
		if _image_area_params is not None:
			return _image_area_params
		area_param_config = ConfigParser()
		section = f"{section_stem}.{app}"
		try:
			with open(area_param_file, encoding="utf8") as rf:
				area_param_config.read_file(rf)
				try:
					_image_area_params = area_param_config[section]
				except KeyError as e:
					logger.warning(
						"Failed to read area parameter file %s: %s", area_param_file, e
					)
					raise ConfigKeyException(
						"Failed to read area parameter section",
						key=section,
						config=area_param_config,
					) from e
		except (TypeError, FileNotFoundError) as e:
			logger.error(
				"Area parameter file is %s: %s",
				"None" if isinstance(e, TypeError) else "not found",
				area_param_file,
			)
			raise ConfigError(
				"Area parameter file is %s: %s"
				% ("None" if isinstance(e, TypeError) else "not found", area_param_file)
			) from e

		except NoSectionError:
			logger.warning("Failed to read area parameter file %s", area_param_file)
			raise  # ConfigError("Failed to read area parameter file %s" % args.area_param_file) from e
		else:
			return _image_area_params

	image_area_params: SectionProxy | None = None
	if args.area_param_file:
		try:
			area_param_config = ConfigParser()
			area_param_config.read(args.area_param_file)
			image_area_params = area_param_config[
				f"image_area_param.{args.app.name.lower()}"
			]
		except Exception as e:
			logger.warning(
				f"Failed to read area parameter file {args.area_param_file}: {e}"
			)
			area_param_config = None
		else:
			logger.info(
				"Area parameter file is read: %s as %s",
				args.area_param_file,
				image_area_params,
			)
	# image_config_filename = (args.file) #.resolve()Path
	# filter_config_doc: TOMLDocument | None = None

	def fill_area_param_dict(
		area_param_dict: dict[ImageAreaParamName, ImageAreaParam] = {},
		image: np.ndarray | None = None,
		exclude_set: set[ImageAreaParamName] = set(),
		y_margin: int = 0
	) -> dict[ImageAreaParamName, ImageAreaParam]:
		for key in exclude_set:
			area_param_dict.pop(key, None)
		for area_name in set(ImageAreaParamName):
			if area_name not in area_param_dict:
				if image is None or image.size == 0:
					logger.error("Image is None or size 0")
					raise ValueError("Image is None or size 0")
				logger.info("Try to get area params from image: %s", image.shape)
				from mouse_event import get_area, QuitKeyException
				try:
					TL, BR = get_area(area_name.name, image)
				except QuitKeyException:
					logger.warning(
						"Failed to get area from image for %s", area_name.name
					)
					continue
				else:
					param_obj = ImageAreaParam(
						TL[1] + y_margin, BR[1] - TL[1], TL[0], BR[0] - TL[0]
					)
					area_param_dict[area_name] = param_obj
		return area_param_dict

	is_file_list_loaded = False

	# from os import scan_dir
	def get_args_files(file_list: list[Path] = []):
		nonlocal is_file_list_loaded
		if not is_file_list_loaded:
			if args.files:
				file_list += [
					Path(f)
					for f in args.files
					if Path(f).is_file()
					and Path(f).suffix in args.image_ext
					and "." + args.app.name.lower() in Path(f).suffixes
				]
				is_file_list_loaded = True
				return file_list
			_file_list = []
			# with scan_dir(args.image_dir) as ee:
			for e in Path(args.image_dir).iterdir():
				if e.is_file and e.suffix in args.image_ext:
					for stem_end in app_to_stem_end_set(args.app):
						if e.stem.endswith(stem_end):
							_file_list.append((e, e.stat().st_mtime))

			file_list += [
				m[0] for m in sorted(_file_list, key=lambda f: f[1], reverse=True)
			]
			is_file_list_loaded = True
			logger.info("Loaded file_list of %d files: %s", len(file_list), file_list)
		return file_list

	def is_wild_card(file_name):
		for c in "*?[]":
			if c in file_name:
				return True
		return False

	image_path_dir: Path | None = None
	# if image_path_dir and not image_path_dir.exists(): sys.exit("Image dir. does not exist: %s" % image_path_dir)

	try:
		_file = get_args_files()[args.nth - 1]
		image_file = image_path_dir / _file if image_path_dir else Path(_file).expanduser()
	except IndexError:
		# image_file = file_list[0]
		sys.exit(f"Index out of range for file_list by {args.nth=}")
	logger.info("Selected file: %s", image_file.name)
	if not image_file.exists():
		sys.exit("Error: image_file not found: %s" % image_file)

	image = cv2.imread(
		str(image_file), cv2.IMREAD_GRAYSCALE
	)  # cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
	if image is None:
		raise ValueError("Error: Could not load image: %s" % image_file)
	bin_image = None
	# check ratios
	is_image_border_ratio_OK = None
	if args.app == APP_NAME.TAIMEE:
		from ocr_filter import OCRFilter

		y_margin, borders, bin_image = OCRFilter.get_borders(image)
		image_border_ratios = OCRFilter.convert_border_offset_ranges_to_ratio_list(
			borders
		)
		# extract border ratio from app_border_ratio
		is_image_border_ratio_OK = True
		for ratio in args.app_border_ratio:
			area_name, v = ratio.split(":")
			if area_name == args.app.name.lower():
				config_border_ratios = [float(i) for i in v.split(",")]
				for n, r in enumerate(config_border_ratios):
					if abs(1 - r / image_border_ratios[n]) > 0.1:
						logger.warning(
							"Warning: image_border_ratio differs significantly from config_border_ratio %s",
							r,
						)
						is_image_border_ratio_OK = False
						# filter_area_param_dict = {}
						logger.info(
							"No use of default filter parameters due to border ratio mismatch for %s",
							args.app.name.lower(),
						)

	try:
		app_filter_class = APP_NAME_TO_FILTER_CLASS[args.app]
	except KeyError:
		from ocr_filter import OCRFilter
		app_filter_class = OCRFilter
		# specific_filter_for_app = False
	# else: specific_filter_for_app = True

	param_dict: dict[ImageAreaParamName, ImageAreaParam] = {}
	section = None
	param_config = None
	# try:
	area_params_section = get_image_area_params_section()
	if is_successful(area_params_section):
		param_str_dict = area_params_section.unwrap()
		for k, v in param_str_dict.items():
			try:
					param = ImageAreaParam.from_str(v)
					param_dict[
				ImageAreaParamName(k)] = param
			except ValueError as e:
				logger.error("Failed to genarate an image area param '%s' obj from filter parameter [%s]: %s", k, v, e)
			else:
				logger.info("Filter parameter for %s is read as: %s", k, param)
	else:
		exception = area_params_section.failure()  # case ConfigKeyException():
		if isinstance(exception, ConfigKeyException):
			assert (
				".".join([args.image_area_param_section_stem + "." + args.app])
				== exception.key
			)
			assert isinstance(exception.config, ConfigParser)
			param_config = exception.config
			logger.warning(
				"Going to get filter parameters manually due to config error for %s",
				args.app.name.lower(),
			)
		else:
			logger.error("Failed to get filter parameters: %s", exception)

	# if not param_dict:
	param_dict = fill_area_param_dict(param_dict, image=bin_image[y_margin:, :], exclude_set=args.exclude_area_param_set) #, y_margin=y_margin)
	section = ".".join([args.image_area_param_section_stem + "." + args.app])
	area_param_config = param_config if param_config is not None else ConfigParser()
	area_param_config[section] = {k: f"{v.param}" for k, v in param_dict.items()}

	# make a function to save the param_config to a config file
	def save_param_dict_atexit():
		try:
			with open(args.area_param_file, "w", encoding="utf8") as wf:
				area_param_config.write(wf)
		except Exception as e:
			logger.warning("Failed to write area parameter file: %s", e)
		else:
			logger.info(
				"Saved `param_config` generated from user input as a file: %s",
				args.area_param_file,
			)

	atexit.register(save_param_dict_atexit)

	app_filter = app_filter_class(
		image=image, param_dict=param_dict, show_check=args.show, bin_image=bin_image, y_offset=-y_margin
	)  # if not args.no_ocr else None

	# if app_filter is not None:
	from tesseract_ocr import TesseractOCR, Output
	from tomli_w import dumps

	ocr = TesseractOCR()
	print(f"# {image_file.name=}")
	month_day: tuple[int, int] | None = None
	doc_dict: dict[
		ImageAreaParamName, str
	] = {}  # newline-sepalated text columns which is tab-separated
	for area_name in ImageAreaParamName: #, area_param in app_filter.param_dict.items():
		try:
			area_param = app_filter.param_dict[area_name]
		except KeyError:
			continue

		ocr_area_name = f"ocr-{area_name.name}"
		# area_tbl = table() area_tbl.add(comment(area_name)) area_tbl.add(nl())
		area_dict = {}
		col_list = []
		print(f"[{ocr_area_name}]")
		for col, ocr_area in enumerate(
			area_param.crop_image(app_filter.image[app_filter.y_margin :, :], 0)#app_filter.y_margin)
		):
			if args.show_ocr_area:
				_plot([ocr_area, app_filter.image[app_filter.y_margin :, :]])
			col_str = f"c{col + 1}"
			result = ocr.exec(ocr_area, output_type=Output.DATAFRAME, psm=args.psm)
			if is_successful(result):
				ocr_result = result.unwrap()
			else:
				logger.warning("Failed to get ocr result: %s", result.failure())
				continue  # ocr_result = None
			max_line = max(ocr_result["line_num"])

			def textline(n, conf=args.ocr_conf):
				return ocr_result[
					(ocr_result["line_num"] == n) & (ocr_result["conf"] > conf)
				]  # ['text'] # return ''.join(

			df_list = []
			for r in range(1, max_line + 1):
				line = textline(r)
				df_list.append(line)  # '\n'.join(line))
			ocr_text_lines = [" ".join(df["text"]) for df in df_list]
			if args.ocr_filter_sqlite_db_name and area_name == ImageAreaParamName.SHIFT:
				# from tool_pyocr import MDateError
				month_day_hours = (
					app_filter_class.extract_month_day_and_hours_from_shift_area_text(
						ocr_text_lines
					)
				)
				if is_successful(month_day_hours):
					month_day, hours = month_day_hours.unwrap()
				else:
					month_day = None
			print(f"{col_str}={ocr_text_lines}")
			col_list.append(ocr_text_lines)
			doc_dict[area_name] = "\n".join(["\t".join(col) for col in col_list])
			# area_tbl.add(comment(pg_str)) area_tbl.add(nl()) area_tbl.add(comment(ocr_text)) area_tbl.add(nl())
		"""if len(col_list) > 1:
			for p, col in enumerate(col_list):
				# col_str = f"p{p+1}"
				area_dict[p] = '\n'.join(col) """
	if args.ocr_filter_sqlite_db_name and month_day is not None:
		try:
			db_fullpath = Path(args.image_dir).expanduser() / args.ocr_filter_sqlite_db_name if args.image_dir else Path(args.ocr_filter_sqlite_db_name).expanduser()
			# database = make_sqlite_db(file=str(db_fullpath.name), folder=str(db_fullpath.parent))
			month: int = month_day.month
			day: int = month_day.day
			year: int = get_data_year(month)
			inserted_item = insert_ocr_data(str(db_fullpath),
				args.app, year, month, day, doc_dict, image_file
			)
		except TypeError as e:
			logger.error("Conversion from str to int failed: %s", e)
		except OperationalError as e:
			logger.error("Database operation failed: %s", e)
		else:
			if inserted_item is not None:
				logger.info("Inserted OCR data [%s] into database: %s", inserted_item, db_fullpath)
		# else: doc_dict[area_name] = '\n'.join(col_list[0])
		# doc.add(area_tbl)
	if args.save:
		save_path = Path(args.save) / f"{image_file.stem}.ocr-{args.app}'.toml'"
		if save_path.exists():
			yn = input(
				f"\nThe file path to save the image file area configuration:{save_path} already exists. Overwrite?(Enter 'Yes' or 'Affirmative' if you want to overwparser.parse_args()rite)"
			).lower()
			if yn != "yes" and yn != "affirmative":
				sys.exit("Exit since the user not accept overwrite of: %s" % save_path)
		toml_text = dumps(
			{f"{area_name}": v for area_name, v in doc_dict.items()},
			multiline_strings=True,
		)
		with save_path.open("w") as wf:
			wf.write(toml_text)
		logger.info("Saved toml file into: %s\n%s", save_path, toml_text)

	if args.make:
		make_path = (
			Path(args.make + ".toml")
			if not args.make.endswith(".toml")
			else Path(args.make)
		)  # Path(args.toml)

		from tomlkit.toml_file import TOMLFile
		from tomlkit import table

		config_file: TOMLFile | None = None
		if make_path.exists():
			try:
				config_file = TOMLFile(make_path)  # get_filter_config()
				org_config = config_file.read()
			except Exception as e:
				logger.error("Failed to load existing TOML file %s: %s", make_path, e)
				raise MakeError(
					f"Failed to load existing TOML file {make_path}: {e}"
				) from e
			else:
				logger.info("config is loaded from: %s", make_path)
		else:
			org_config = TOMLDocument()
			logger.info("config is created")
		# from io import StringIO
		# sio = StringIO()
		# with make_path.open('w') as wf: # dump(doc, wf)
		# label = f"[ocr-filter.{str(app_name)}]"
		# print(label, file=wf)
		# print(f"[ocr-filter.{label}]", file=wf)
		from tomlkit import container as TKContainer

		ocr_filter_table = (
			org_config.get(OCR_FILTER)
			or org_config.add(OCR_FILTER, table())[OCR_FILTER]
		)
		org_area_dict: dict = ocr_filter_table[args.app] if ocr_filter_table else {}
		for key, param in app_filter.param_dict.items():
			different = False
			area_name = key.name.lower()
		# logger.info("Image area parameters are saved into %s\nas: %s", make_path, sio.read())
		# print("[ocr-filter.taimee]")
		# print(sio.read())
	# --toml ocr-filter
	"""[ocr-filter.taimee]
HeadingAreaParam = [0, 111, 196, -1]
ShiftAreaParam = [219, 267, 345, 373]
BreaktimeAreaParam = [488, 224, 0, 720]
PaystubAreaParam = [714, -1, 0, -1]"""
database_environ_str = 'OCR_FILTER_SQLITE_DB_PATH'
def insert_ocr_data(database_path:str, app: APP_NAME, year: int, month: int, day: int, data: dict[ImageAreaParamName, str], file: Path, hours:Sequence[str]|None=None):
	from os import environ
	environ[database_environ_str] = database_path
	from ocr_filter_model import database
	if not database.is_connection_usable():
		database.connect()
	from ocr_filter_model import App, ImageRoot, PaystubOCR
	App.create_table(safe=True)
	ImageRoot.create_table(safe=True)
	PaystubOCR.create_table(safe=True)
	app_obj, created = App.get_or_create(name=app)
	if created:
		logger.info("Created app: %s as app_obj: %s", app, app_obj)
	resolved_root = str(file.parent.resolve()) 
	root_obj, created= ImageRoot.get_or_create(root=resolved_root)
	if created:
		logger.info("Created root: %s as root_obj: %s", resolved_root, root_obj)
	# except ImageRoot.DoesNotExist: root_model = ImageRoot.create(root=resolved_root)
	old_item = PaystubOCR.get_or_none(app==app_obj, year==year, month==month, day==day)
	if old_item is None: # if not old_item:
		from ocr_filter_model import get_file_checksum_md5
		checksum = get_file_checksum_md5(file)
		from datetime import datetime as Datetime
		new_item, created = PaystubOCR.get_or_create(
			app=app_obj,
			year=year,
			month=month,
			day=day,
			defaults = {
			'modified_at':Datetime.now(),
			'heading_text':data.get(ImageAreaParamName.HEADING),
			'shift_text':data.get(ImageAreaParamName.SHIFT),
			'breaktime_text':data.get(ImageAreaParamName.BREAKTIME),
			'paystub_text':data.get(ImageAreaParamName.PAYSTUB),
			'salary_text':data.get(ImageAreaParamName.SALARY),
			'root':root_obj,
			'file':file.name,
			'checksum':checksum,
			}
		)
		if created:
			logger.info("New record is created:%s", new_item)
		else:
			logger.info("Old record is updated:%s", new_item)
		database.commit()
		return new_item
if __name__ == "__main__":
	try:
		main()
	except SystemExit as e:
		if e.code != 0:
			logger.error("SystemExit: %s", e)
			raise