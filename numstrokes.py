from typing import Callable, Iterator
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
from strok7 import SEGPATH_SLANT, get_segpath_for_c, Sp
from seg_7_digits import homo_seg_7_array
from seven_seg import SEVEN_SEG_SIZE

NUMSTROKE_SLANT = SEGPATH_SLANT

segpath_list = [None] * len(homo_seg_7_array)

for i, segs in enumerate(homo_seg_7_array):
	spth_list = []
	for c in segs:
		if c:
			path_pair: tuple[Sp,Sp] = get_segpath_for_c(c).path
			spth_list.append(path_pair)
	segpath_list[i] = spth_list
class NumStrokes:
	f'''strokes[{SEVEN_SEG_SIZE}]: slanted strokes]'''
	def __init__(self, scale: float=1.0, offset: tuple[int, int]=(0, 0), slant=NUMSTROKE_SLANT, max_cache=len(homo_seg_7_array)):
		self.scale = scale
		self.offset = offset
		self.slant = slant
		self.strokes: Callable[[int], NDArray] = lru_cache(maxsize=max_cache)(self._strokes)

	def _strokes(self, n: int)-> NDArray:
		stroke_list = [path.slant(s=self.slant, scale=self.scale, offset=self.offset) for path in segpath_list[n]]
		return stroke_list

def make_basic_segpath_list(segpath_list = [], homo=True):
		for i, segs in enumerate(homo_seg_7_array):
			spth_list = []
			for c in segs:
				if c:
					path: tuple[Sp,Sp] = get_segpath_for_c(c).path
					spth_list.append(path)
				else:
					if homo:
						spth_list.append([(0,0),(0,0)])
			segpath_list.append(spth_list)
		return segpath_list

def make_basic_strokes(segpath_list = [], homo=True):
		for i, segs in enumerate(homo_seg_7_array):
			spth_list = []
			for c in segs:
				if c:
					path: tuple[Sp,Sp] = get_segpath_for_c(c).path
					path_list = [pt.xy for pt in path]
					spth_list.append(path_list)
				else:
					if homo:
						spth_list.append([(0,0),(0,0)])
			segpath_list.append(spth_list)
		return segpath_list

class SlantedNumStrokes(NumStrokes):
	def __init__(self, max_cache=SEVEN_SEG_SIZE, scale = 1, offset = (0, 0),slant=SEGPATH_SLANT):
		super().__init__(scale, offset)
		self.slant: float = slant
		self.strokes: Callable[[int], NDArray] = lru_cache(maxsize=max_cache)(self._slanted_strokes)
	def _slanted_strokes(self, n: int)-> list[tuple[int, int]]:
		strokes = [my_round2(st * self.scale + self.offset) for st in super().strokes(n)]
		return strokes

def my_round2(x, decimals=0):
    return np.sign(x) * np.floor(np.abs(x) * 10**decimals + 0.5) / 10**decimals

if __name__ == '__main__':
	from pprint import pp
	import sys

	num_strokes = NumStrokes()
	for i in range(SEVEN_SEG_SIZE):
		stroke = num_strokes.strokes(i)
		pp(stroke)
	num_strokes = SlantedNumStrokes()
	for i in range(SEVEN_SEG_SIZE):
		stroke = num_strokes.strokes(i)
		pp(stroke)
	basic_segpath_list = make_basic_segpath_list(homo=False)
	pp(basic_segpath_list)
	basic_strokes = make_basic_strokes(homo=False)
	pp(basic_strokes)