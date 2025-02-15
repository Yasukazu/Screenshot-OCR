import pickle
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

class NumStrokes:
	def __init__(self, slant=0.25):
		self._strokes = [get_num_strokes(n, slant) for n in range(SEVEN_SEG_SIZE)]

	@property
	def strokes(self):#, n: int):
		#assert 0 <= n < SEVEN_SEG_SIZE
		return self._strokes #[n]

import numpy as np
from PIL import ImageDraw
def draw_num(n: int, drw: ImageDraw, offset=(0,0), scale=16, width=8, fill=(0,), strokes=NumStrokes(0.25).strokes):
	'''draw number as 7-segment digital: 0 to 9 makes [0123456789], 10 to 15 makes [ABCDEF], 16 makes hyphen(-)'''
	assert 0 <= n < SEVEN_SEG_SIZE
	if not isinstance(offset, np.ndarray): #, npt.generic)):
		offset = np.array(offset, int)
	for stroke in strokes[n]: #get_num_strokes(n, slant=slant):
			strk = np.array(stroke, np.float16)
			seq = [tuple(st * scale + offset) for st in strk]
			drw.line(seq, fill=fill, width=width)

if __name__ == '__main__':
	import sys
	from pprint import pp
	from PIL import Image
	save = False
	show_list = [SEVEN_SEG_SIZE - 1]
	scale = 40
	offset = np.array([20, 20], int)
	strokes = NumStrokes(0.25).strokes
	for i in show_list:
		img = Image.new('L', (80, 160), (0xff,))
		drw = ImageDraw.Draw(img)
		draw_num(i, drw, offset=offset, scale=scale, width=8, strokes=strokes)
		img.show()
		if save:
			img.save(f"digi-{i}.png", 'PNG')	
	sys.exit(0)
	sgp_array = load_segpath_array()
	seg_list = get_segpath_list()
	from pprint import pprint
	pp(seg_list[0])