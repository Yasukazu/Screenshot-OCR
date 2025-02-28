from typing import Callable, Iterator
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
from strok7 import SEGPATH_SLANT, get_segpath_for_c, Sp0
from seg_7_digits import seg_7_array, homo_seg_7_array
from seven_seg import SEVEN_SEG_SIZE

NUMSTROKE_SLANT = SEGPATH_SLANT

def get_segpath_list(segpath_list = [None] * len(seg_7_array)):
	for i, segs in enumerate(seg_7_array):
		spth_list = []
		for c in segs:
			if c:
				path_pair: tuple[Sp0,Sp0] = get_segpath_for_c(c).path
				spth_list.append(path_pair)
		segpath_list[i] = spth_list
	return segpath_list	

class BasicNumStrokes:
	f'''strokes[{SEVEN_SEG_SIZE}]: slanted strokes]'''
	_segpath_list = get_segpath_list()
	def __init__(self, max_cache=2 * SEVEN_SEG_SIZE):
		self.strokes: Callable[[int], list] = lru_cache(maxsize=max_cache)(self._strokes)

	def _strokes(self, n: int, slant=0, scale=1, offset=(0, 0)):
		stroke_list = [(p1.slant(s=slant, scale=scale, offset=offset),
		p2.slant(s=slant, scale=scale, offset=offset),) for (p1, p2) in self._segpath_list[n]]
		return stroke_list



class NumStrokes:
	f'''strokes[{SEVEN_SEG_SIZE}]: slanted strokes]'''
	def __init__(self, scale: float=1.0, offset: tuple[int, int]=(0, 0), slant=NUMSTROKE_SLANT, max_cache=len(homo_seg_7_array)):
		self.scale = scale
		self.offset = offset
		self.slant = slant
		self.strokes: Callable[[int], NDArray] = lru_cache(maxsize=max_cache)(self._strokes)

	def _strokes(self, n: int)-> NDArray:
		stroke_list = [(p1.slant(s=self.slant, scale=self.scale, offset=self.offset),
		p2.slant(s=self.slant, scale=self.scale, offset=self.offset)) for (p1, p2) in segpath_list[n]]
		return stroke_list

if __name__ == '__main__':
	from pprint import pp
	offset=(7,3)
	scale=2
	num_strokes = BasicNumStrokes()
	for i in range(SEVEN_SEG_SIZE):
		stroke = num_strokes.strokes(i, scale=scale, offset=offset)
		pp(stroke)

