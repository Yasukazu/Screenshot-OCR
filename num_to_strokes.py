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

from seg_7_digits import get_seg_7_list, seg_7_array

def load_segpath_array_b(segelem_dict=get_segelem_dict(), _segpath_array = [None] * SEVEN_SEG_SIZE, segment_array: tuple[tuple[str]]=seg_7_array)-> list[list[segpath_dict]]:
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

'''def _load_seg7(pkl: str=SEVEN_SEG_STEM + PICKLE_EXT):
	rf = open(pkl, 'rb').read()
	csv_reader = pickle.loads(rf)
	for i, row in enumerate(csv_reader):
		seg = Seg7(*row)
		assert seg.h == i
		_seg7_array.append(seg)
	assert len(_seg7_array) == 16
	return _seg7_array

assert len(_seg7_array) == 16'''


def get_seg_lines(n):
	seg_line_set = _seg7_array[n]
	return [line.value for line in seg_line_set]

from numstrokes import NumStrokes, NUMSTROKE_SLANT

def get_num_strokes(n: int, slant=NUMSTROKE_SLANT, segpath_list: list[list[segpath_dict]]=get_segpath_list())-> list[list[tuple[float, float]]]:
	if not 0 <= n < SEVEN_SEG_SIZE:
		raise ValueError(f"{n} is out of range.")
	segpath: list[segpath_dict] = segpath_list[n]
	return [path.slanted(slant) for path in segpath]

class ImageFill(Enum): # single element tuple for ImageDraw color
	BLACK = (0,)
	WHITE = (0xff,)
	@classmethod
	def invert(cls, fill):
		if fill == ImageFill.BLACK:
			return ImageFill.WHITE
		return ImageFill.BLACK
from collections import namedtuple
Size = namedtuple('Size', ['w', 'h'])
def get_number_image(size: Size, nn: Sequence[int | FormatNum] | bytearray, bgcolor=ImageFill.WHITE, padding=0.4)-> tuple[Image.Image, Size]: #, slant=0.25, padding=0.2):
	'''returns Image, margins'''
	from format_num import formatnums_to_bytearray
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

def draw_digit(n: int, img: Image.Image, offset: np.ndarray | tuple[int, int]=(0,0), scale=16, width_ratio=0.2, fill=ImageFill.BLACK, stroke_feeder=NumStrokes(), feeder_params=False):
	'''draw a digit as 7-segment shape: 0 to 9 makes [0123456789], 10 to 15 makes [ABCDEF], 16 makes hyphen(-)'''
	assert 0 <= n < SEVEN_SEG_SIZE
	drw = ImageDraw.Draw(img)
	width = int(scale * width_ratio) or 1
	if feeder_params:
		for seq in stroke_feeder.strokes(n):
			jseq = [(int(i), int(j)) for i, j in seq]
			drw.line(jseq, fill=fill.value, width=width)
	else:
		if not isinstance(offset, np.ndarray): #, npt.generic)):
			offset = np.array(offset, int)
		strokes = stroke_feeder.pure_strokes(n)
		for stroke in strokes: # np.array((stroke), np.float16)
				seq = [my_round2(st * scale + offset) for st in stroke]
				jseq = [(int(i), int(j)) for i, j in seq]
				drw.line(jseq, fill=fill.value, width=width)

from functools import wraps
import numpy as np
from path_feeder import PathFeeder
from enum import IntEnum
class AddPos(IntEnum):
	L = -1
	C = 0
	R = 1

def add_number(size: tuple[int, int]=(100, 50), pos: AddPos=AddPos.C, bgcolor=ImageFill.WHITE): # tuple[int, int]=(0, 0)):
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
					case AddPos.L:
						margin_list[0] = 0
					case AddPos.R:
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
	image_size = (80, 40)
	stroke_feeder = NumStrokes(scale=8, offset=(20, 10))
	for n in range(17):
		img = Image.new('L', image_size, (0xff,))
		draw_digit(n, img, stroke_feeder=stroke_feeder, feeder_params=True)
		img.show()
	sys.exit(0)
	a_list = load_segpath_array()
	b_list = load_segpath_array_b()
	num_strokes=[list(stroke) for stroke in NumStrokes(slant=NUMSTROKE_SLANT, segpath_list=b_list).strokes]
	import sys
	for n in range(17):
		draw_digit(n, num_strokes=num_strokes)
		break
	sys.exit(0)
	from pprint import pp
	path_feeder = PathFeeder()

	@add_number(size=(90, 45), pos=AddPos.R)
	def img_open(stem: str):
		fullpath = path_feeder.dir / (stem + path_feeder.ext)
		if fullpath.exists():
			img = Image.open(fullpath)
			return img

	embed_img = img_open('02A1')
	embed_img.show()
	sys.exit(0)
	first_fullpath = path_feeder.first_fullpath
	item_img = Image.open(first_fullpath)
	first_stem = first_fullpath.stem
	name_num_array = [int(c) for c in first_stem]
	num_img, margins = get_number_image((100, 50), name_num_array)
	item_img.paste(num_img, tuple(np.array(margins) + np.array([0, 10])))
	item_img.show()
	sys.exit(0)
	save = False
	show_list = [SEVEN_SEG_SIZE - 1]
	scale = 40
	offset = np.array([20, 20], int)
	strokes = NumStrokes(0.25).strokes
	for i in show_list:
		img = Image.new('L', (80, 160), (0xff,))
		drw = ImageDraw.Draw(img)
		draw_digit(i, drw, offset=offset, scale=scale, width_ratio=8, num_strokes=strokes)
		img.show()
		if save:
			img.save(f"digi-{i}.png", 'PNG')	
	sys.exit(0)
	sgp_array = load_segpath_array()
	seg_list = get_segpath_list()
	from pprint import pprint
	pp(seg_list[0])