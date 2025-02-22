from enum import Enum
import pickle
import numpy as np
from PIL import ImageDraw, Image
from numpy.typing import NDArray
from strok7 import STRK_DICT_STEM

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

from strok7 import SegElem, _SEGELEM7DICT
_segelem_array: list[tuple[Seg7]] = [set()] * 16

from seven_seg import SEVEN_SEG_SIZE, load_7_seg_num_csv_as_df

def load_segpath_array(_segpath_array = [None] * SEVEN_SEG_SIZE, df: NDArray=load_7_seg_num_csv_as_df()):
	'''call "slanted" for each element'''
	for i in range(SEVEN_SEG_SIZE):
		elem_list = []
		for c in 'abcdefg':
			if (df[c])[i] > 0:
				segpath = _SEGELEM7DICT[c]
				elem_list.append(segpath)
		_segpath_array[i] = elem_list
	return _segpath_array

_segpath_array = load_segpath_array()

def get_segpath_list():
	return _segpath_array

from strok7 import SegLine

_seg7_array: list[set[Seg7]] = [set()] * SEVEN_SEG_SIZE
def load_seg7(df: NDArray=load_7_seg_num_csv_as_df())-> list[set[SegLine]]:
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

def get_num_strokes(n: int, slant=0.25):
	assert 0 <= n < SEVEN_SEG_SIZE
	segpath = get_segpath_list()[n]
	for path in segpath:
		yield list(path.slanted(slant))

from functools import cached_property
from collections.abc import Sequence
class NumStrokes:
	def __init__(self, slant=0.25):
		self._strokes = [get_num_strokes(n, slant) for n in range(SEVEN_SEG_SIZE)]

	@property
	def strokes(self)-> Sequence:#, n: int):
		#assert 0 <= n < SEVEN_SEG_SIZE
		return self._strokes #[n]

from format_num import FormatNum, HexFormatNum, conv_num_to_bin

class ImageFill(Enum):
	BLACK = (0,)
	WHITE = (0xff,)
	@classmethod
	def invert(cls, fill):
		if fill == ImageFill.BLACK:
			return ImageFill.WHITE
		return ImageFill.BLACK
from collections import namedtuple
Size = namedtuple('Size', ['w', 'h'])
def get_number_image(size: Size, nn: Sequence[int | FormatNum], bgcolor=ImageFill.WHITE, padding=0.4)-> tuple[Image.Image, Size]: #, slant=0.25, padding=0.2):
	'''returns Image, margins'''
	b_str = []
	for n in nn:
		b_s = n.conv_to_bin() if isinstance(n, FormatNum) else conv_num_to_bin(n)
		b_str.extend(b_s)
	def scale_margins()-> tuple[int, Size]:
		'''returns scale, margin-size'''
		img_ratio = len(b_str) / 2
		w = size[0]
		h = size[1]
		win_ratio = w / h
		x_margin = y_margin = 0
		if win_ratio > img_ratio:
			scale = h / 2
			x_margin = (w - h * img_ratio) / 2
			return int(scale) or 1, Size(int(x_margin), 0)
		scale = w / len(b_str)
		y_margin = (h - w / img_ratio) / 2
		return int(scale) or 1, Size(0, int(y_margin))
	scale, margins = scale_margins()
	font_scale = int(scale * (1 - padding))
	img_size = np.array(tuple(Size(w=len(b_str) * scale * (1 + (scale / font_scale) / scale), h=2 * scale)))
	offset = (np.array([scale, 2 * scale]) - np.array([font_scale, 2 * font_scale])) / 2
	pitch = np.array([scale, 0])
	img_tuple = list(int(i) for i in img_size)
	img = Image.new('L', img_tuple, color=bgcolor.value)
	drw = ImageDraw.Draw(img)
	for i in range(len(b_str)): # offset in enumerate(ws.offsets):
		d = int(b_str[i])
		draw_digit(d, drw, i * pitch + offset, scale=font_scale, width_ratio=0.4, fill=bgcolor.invert(bgcolor))
	return img, margins


def draw_digit(n: int, drw: ImageDraw, offset: np.ndarray | tuple[int, int]=(0,0), scale=16, width_ratio=0.2, fill=ImageFill.BLACK, strokes=NumStrokes(0.25).strokes):
	'''draw a digit as 7-segment shape: 0 to 9 makes [0123456789], 10 to 15 makes [ABCDEF], 16 makes hyphen(-)'''
	assert 0 <= n < SEVEN_SEG_SIZE
	width = int(scale * width_ratio) or 1
	if not isinstance(offset, np.ndarray): #, npt.generic)):
		offset = np.array(offset, int)
	for stroke in strokes[n]: #get_num_strokes(n, slant=slant):
			strk = np.array(stroke, np.float16)
			seq = [tuple(st * scale + offset) for st in strk]
			drw.line(seq, fill=fill.value, width=width)

from functools import wraps
import numpy as np
from path_feeder import PathFeeder
def add_number(size: tuple[int, int]=(100, 50)):
	def _embed_number(func):
		@wraps(func)
		def wrapper(*ag, **kw):
			item_img = func(*ag, **kw)
			if item_img:
				name_num_array = [int(c) for c in ag[0]]
				num_img, margins = get_number_image(size, name_num_array)
				item_img.paste(num_img, tuple(np.array(margins) + np.array([0, 10])))
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
	import sys
	from pprint import pp
	path_feeder = PathFeeder()

	@add_number(size=(90, 45))
	def img_open(stem: str):
		fullpath = path_feeder.dir / (stem + path_feeder.ext)
		if fullpath.exists():
			img = Image.open(fullpath)
			return img

	embed_img = img_open('01')
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
		draw_digit(i, drw, offset=offset, scale=scale, width_ratio=8, strokes=strokes)
		img.show()
		if save:
			img.save(f"digi-{i}.png", 'PNG')	
	sys.exit(0)
	sgp_array = load_segpath_array()
	seg_list = get_segpath_list()
	from pprint import pprint
	pp(seg_list[0])