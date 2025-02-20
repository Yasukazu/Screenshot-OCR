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

def get_number_image(ht: int, wt: int, nn: Sequence[int | FormatNum], bgcolor=ImageFill.WHITE): #, slant=0.25, padding=0.2):
	b_str = []
	for n in nn:
		b_s = n.conv_to_bin() if isinstance(n, FormatNum) else conv_num_to_bin(n)
		b_str.extend(b_s)
	from wh_solve import solve_wh
	ws = solve_wh(ht=ht, n=len(b_str)) # WHSolve(width, len(b_str))
	img = Image.new('L', ws.box_size, color=bgcolor.value)
	drw = ImageDraw.Draw(img)
	for i, offset in enumerate(ws.offsets):
		d = int(b_str[i])
		draw_digit(d, drw, offset, ws.scale, fill=bgcolor.invert(bgcolor))
	return img


def draw_digit(n: int, drw: ImageDraw, offset=(0,0), scale=16, width_ratio=0.2, fill=ImageFill.BLACK, strokes=NumStrokes(0.25).strokes):
	'''draw a digit as 7-segment shape: 0 to 9 makes [0123456789], 10 to 15 makes [ABCDEF], 16 makes hyphen(-)'''
	assert 0 <= n < SEVEN_SEG_SIZE
	width = int(scale * width_ratio) or 1
	if not isinstance(offset, np.ndarray): #, npt.generic)):
		offset = np.array(offset, int)
	for stroke in strokes[n]: #get_num_strokes(n, slant=slant):
			strk = np.array(stroke, np.float16)
			seq = [tuple(st * scale + offset) for st in strk]
			drw.line(seq, fill=fill.value, width=width)

if __name__ == '__main__':
	import sys
	from pprint import pp
	hgt = 30
	wdt = 80
	img = get_number_image(hgt, wdt, [24, HexFormatNum(-0xa)], bgcolor=ImageFill.BLACK)
	img.show()
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