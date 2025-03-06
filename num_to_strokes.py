import sys
from collections.abc import Callable
from functools import cached_property
from collections.abc import Sequence
from enum import Enum
import pickle
import numpy as np
from PIL import ImageDraw, Image
from numpy.typing import NDArray
from pandas import DataFrame
from strok7 import STRK_DICT_STEM
from format_num import FormatNum, HexFormatNum, conv_num_to_bin
from digit_image import ImageFill

def conv(i: str)-> bool:
	return False if i == '' else bool(int(i))
class Seg7:
	def __init__(self, a: str, b: str, c: str, d: str, e: str, f: str, g: str, h: str):
		self.a: bool = conv(a)
		self.b: bool = conv(b)
		self.c: bool = conv(c)
		self.d: bool = conv(d)
		self.e: bool = conv(e)
		self.f: bool = conv(f)
		self.g: bool = conv(g)
		self.h: bool = conv(h)

SEVEN_SEG_STEM = '7-seg'
PICKLE_EXT = '.pkl'

from strok7 import SegPath, get_segelem_dict, get_segpath_for_c
_segelem_array: list[tuple[Seg7]] = [set()] * 16

from seven_seg import SEVEN_SEG_SIZE, load_7_seg_num_pkl # load_7_seg_num_csv_as_df
type segpath_dict = dict[str, SegPath]
def load_segpath_array(segelem_dict=get_segelem_dict(), _segpath_array = [None] * SEVEN_SEG_SIZE, df: DataFrame=load_7_seg_num_pkl())-> list[list[segpath_dict]]:
	'''call "slanted" for each element'''
	if not _segpath_array[0]:
		for i in range(SEVEN_SEG_SIZE):
			elem_list: list[dict[str, SegPath]] = []
			for c in 'abcdefg':
				if (df[c])[i] > 0:
					segpath: dict[str, SegPath] = segelem_dict[c]
					elem_list.append(segpath)
			_segpath_array[i] = elem_list
	return _segpath_array

from seg_7_digits import get_seg_7_list, homo_seg_7_array

def load_segpath_array_b(segelem_dict=get_segelem_dict(), _segpath_array = [None] * SEVEN_SEG_SIZE, segment_array: tuple[tuple[str]]=homo_seg_7_array)-> list[list[segpath_dict]]:
	'''call "slanted" for each element'''
	if not _segpath_array[0]:
		for i, df in enumerate(segment_array):
			_segpath_array[i] = [get_segpath_for_c(c) for c in df]
	return _segpath_array

def get_segpath_list(load_func: Callable[[], list[list[segpath_dict]]] = load_segpath_array)-> list[list[segpath_dict]]:
	return load_func()

from strok7 import SegLine

_seg7_array: list[set[Seg7]] = [set()] * SEVEN_SEG_SIZE
def load_seg7(df: DataFrame=load_7_seg_num_pkl())-> list[set[SegLine]]:
	for i in range(SEVEN_SEG_SIZE):
		dline = df[i: i + 1]
		seg_set = set()
		for c in 'abcdefg':
			if int(list(dline[c])[0]) > 0:
				seg = SegLine.get(c)
				seg_set.add(seg) # dlst.append(seg) # '1') # SegFlag.get(c).value)
		_seg7_array[i] = seg_set
	return _seg7_array
def get_seg7_list():
	return _seg7_array



def get_seg_lines(n):
	seg_line_set = _seg7_array[n]
	return [line.value for line in seg_line_set]

from num_strokes import NumStrokes, NUMSTROKE_SLANT, DigitStrokes #, SlantedNumStrokes

def get_num_strokes(n: int, slant=NUMSTROKE_SLANT, segpath_list: list[list[segpath_dict]]=get_segpath_list())-> list[list[tuple[float, float]]]:
	if not 0 <= n < SEVEN_SEG_SIZE:
		raise ValueError(f"{n} is out of range.")
	segpath: list[segpath_dict] = segpath_list[n]
	return [path.slanted(slant) for path in segpath]

from collections import namedtuple
Size = namedtuple('Size', ['w', 'h'])
from format_num import formatnums_to_bytearray
from digit_image import BasicDigitImage
def get_basic_number_image(nn: Sequence[int | FormatNum] | bytearray, digit_image_feeder=BasicDigitImage())-> Image.Image:
	b_array = nn if isinstance(nn, bytearray) else formatnums_to_bytearray(nn)
	number_image_size = len(b_array) * digit_image_feeder.size[0], digit_image_feeder.size[1]
	number_image = Image.new('L', number_image_size, (0,))
	offset = (0, 0)
	x_offset = digit_image_feeder.size[0]
	for n in b_array:
		digit_image = digit_image_feeder.get(n)
		number_image.paste(digit_image, offset)
		offset = offset[0] + x_offset, 0
	return number_image

def get_number_image(size: Size, nn: Sequence[int | FormatNum] | bytearray, bgcolor=ImageFill.WHITE, padding=0.4)-> tuple[Image.Image, Size]: #, slant=0.25, padding=0.2):
	'''returns Image, margins'''
	b_array = nn if isinstance(nn, bytearray) else formatnums_to_bytearray(nn)

	def scale_margins()-> tuple[int, Size]:
		'''returns scale, margin-size'''
		img_ratio = len(b_array) / 2
		w = size[0]
		h = size[1]
		win_ratio = w / h
		x_margin = y_margin = 0
		if win_ratio > img_ratio:
			scale = h / 2
			x_margin = (w - h * img_ratio) / 2
			return int(scale) or 1, Size(int(x_margin), 0)
		scale = w / len(b_array)
		y_margin = (h - w / img_ratio) / 2
		return int(scale) or 1, Size(0, int(y_margin))
	scale, margins = scale_margins()
	font_scale = int(scale * (1 - padding))
	img_size = np.array(tuple(Size(w=len(b_array) * scale * (1 + (scale / font_scale) / scale), h=2 * scale)))
	offset = (np.array([scale, 2 * scale]) - np.array([font_scale, 2 * font_scale])) / 2
	pitch = np.array([scale, 0])
	img_tuple = list(int(i) for i in img_size)
	img = Image.new('L', img_tuple, color=bgcolor.value)
	drw = ImageDraw.Draw(img)
	for i in range(len(b_array)): # offset in enumerate(ws.offsets):
		d = (b_array[i])
		draw_digit(d, drw, i * pitch + offset, scale=font_scale, width_ratio=0.4, fill=bgcolor.invert(bgcolor))
	return img, margins

def my_round2(x, decimals=0):
    return np.sign(x) * np.floor(np.abs(x) * 10**decimals + 0.5) / 10**decimals

def draw_digit(n: int, img: Image.Image, offset: np.ndarray | tuple[int, int]=(0,0), scale=16, width_ratio=0.2, fill=ImageFill.BLACK, stroke_feeder=DigitStrokes(), feeder_params=False):
	'''draw a digit as 7-segment shape: 0 to 9 makes [0123456789], 10 to 15 makes [ABCDEF], 16 makes hyphen(-)'''
	assert 0 <= n < SEVEN_SEG_SIZE
	def scale_offset(v: float, pos: int)-> int:
		return round(scale * v + offset[pos])
	drw = ImageDraw.Draw(img)
	width = int(scale * width_ratio) or 1
	if isinstance(stroke_feeder, DigitStrokes):
		for seq in stroke_feeder.strokes(n): #, scale=scale, offset=offset):
			jseq = [(scale_offset(i, 0), scale_offset(j, 1)) for i, j in seq]
			drw.line(jseq, fill=fill.value, width=width)
	else:
		if not isinstance(offset, np.ndarray): #, npt.generic)):
			offset = np.array(offset, int)
		strokes = stroke_feeder.pure_strokes(n)
		for stroke in strokes: # np.array((stroke), np.float16)
				seq = [my_round2(st * scale + offset) for st in stroke]
				jseq = [(int(i), int(j)) for i, j in seq]
				drw.line(jseq, fill=fill.value, width=width)

def get_digit_strokes(n: int, offset: np.ndarray | tuple[int, int]=(0,0), scale=1, stroke_feeder=NumStrokes(), feeder_params=False):
	'''to get strokes to draw a digit as 7-segment shape: 0 to 9 makes [0123456789], 10 to 15 makes [ABCDEF], 16 makes hyphen(-)'''
	assert 0 <= n < SEVEN_SEG_SIZE
	if feeder_params:
		return stroke_feeder.strokes(n)
	else:
		if not isinstance(offset, np.ndarray): #, npt.generic)):
			offset = np.array(offset, int)
		strokes = stroke_feeder.pure_strokes(n)
		return [my_round2(st * scale + offset) for st in [stroke for stroke in strokes]]

from functools import wraps
import numpy as np
from path_feeder import PathFeeder
from enum import IntEnum
class PutPos(IntEnum):
	L = -1
	C = 0
	R = 1

from digit_image import BasicDigitImage
from format_num import HexFormatNum

def put_number(pos: PutPos=PutPos.L, digit_image_feeder=BasicDigitImage()):
	from format_num import HexFormatNum
	def _embed_number(func):
		@wraps(func)
		def wrapper(*ag, **kw):
			item_img: Image.Image = func(*ag, **kw)
			if item_img and kw['number_str']:
				name_num_array = HexFormatNum.str_to_bin(kw['number_str'])
				num_img = get_basic_number_image(name_num_array, digit_image_feeder=digit_image_feeder)
				x_offset = 0
				match pos:
					case PutPos.R:
						x_offset = item_img.width - num_img.width
				item_img.paste(num_img, (x_offset, 0))
				return item_img
		return wrapper
	return _embed_number

def add_number(size: tuple[int, int]=(100, 50), pos: PutPos=PutPos.C, bgcolor=ImageFill.WHITE, stroke_feeder=None): # tuple[int, int]=(0, 0)):
	from format_num import HexFormatNum
	def _embed_number(func):
		@wraps(func)
		def wrapper(*ag, **kw):
			item_img = func(*ag, **kw)
			if item_img and kw['number_str']:
				name_num_array = HexFormatNum.str_to_bin(kw['number_str'])
				num_img, margins = get_number_image(size, name_num_array, bgcolor=bgcolor)
				margin_list = list(margins)
				match pos:
					case PutPos.L:
						margin_list[0] = 0
					case PutPos.R:
						margin_list[0] = item_img.width - num_img.width
				item_img.paste(num_img, [int(v) for v in margin_list] )
				return item_img
		return wrapper
	return _embed_number
def embed_number(func):
	@wraps(func)
	def wrapper(*ag, **kw):
		item_img = func(*ag, **kw)
		if item_img:
			name_num_array = [int(c) for c in ag[0]]
			num_img, margins = get_number_image((100, 50), name_num_array)
			item_img.paste(num_img, tuple(np.array(margins) + np.array([0, 10])))
			return item_img
	return wrapper

if __name__ == '__main__':
	from get_image_outline import detect_tim_top_left_edge
	digit_image = BasicDigitImage(scale=24, line_width=6, padding=(4, 4))
	number_image = get_basic_number_image([0, 2], digit_image_feeder=digit_image)
	from path_feeder import PathFeeder
	feeder = PathFeeder()
	for fe in feeder.feed():
		fullpath = feeder.dir / (fe + feeder.ext)
		image = Image.open(fullpath)
		edge = detect_tim_top_left_edge(image)
		image.paste(number_image, (0, 0))
		breakpoint()
	sys.exit(0)
	image_size = (80, 40)
	stroke_feeder = DigitStrokes()
	stroke_dict = {}
	for n in range(17):
		img = Image.new('L', image_size, (0xff,))
		drw = ImageDraw.Draw(img)
		strokes = stroke_feeder.strokes(n, slant=0.2, scale=8, offset=(20, 10))
		for stroke in strokes:
			drw.line(stroke)
		img.show()
		# strks = get_digit_strokes(n, stroke_feeder=stroke_feeder, feeder_params=True)
		# stroke_dict[n] = strks
		'''for seq in strks:
			jseq = [(int(i), int(j)) for i, j in seq]
			if jseq[0] != jseq[1]:
				drw.line(jseq)
		draw_digit(n, img, stroke_feeder=stroke_feeder, feeder_params=True)'''
