from typing import Callable, Sequence
from types import MappingProxyType
from functools import lru_cache
from numpy.typing import NDArray
from strok7 import SEG_POINT_ARRAY, SEGPATH_SLANT, get_segpath_for_c, Sp, StrokeSlant, i_i_tpl, SpPair
from seg_7_digits import Seg7Bit8, SEG7_DIGIT_ARRAY, seg_7_array, homo_seg_7_array
from seven_seg import SEVEN_SEG_SIZE

SEG7_TO_POINTS = MappingProxyType({
	Seg7Bit8.A: (SEG_POINT_ARRAY[0], SEG_POINT_ARRAY[1]),
	Seg7Bit8.B: (SEG_POINT_ARRAY[1], SEG_POINT_ARRAY[2]),
	Seg7Bit8.C: (SEG_POINT_ARRAY[2], SEG_POINT_ARRAY[3]),
	Seg7Bit8.D: (SEG_POINT_ARRAY[3], SEG_POINT_ARRAY[4]),
	Seg7Bit8.E: (SEG_POINT_ARRAY[4], SEG_POINT_ARRAY[5]),
	Seg7Bit8.F: (SEG_POINT_ARRAY[5], SEG_POINT_ARRAY[0]),
	Seg7Bit8.G: (SEG_POINT_ARRAY[5], SEG_POINT_ARRAY[2]),
}) # : dict[Seg7, tuple[Sp, Sp]]

NUMSTROKE_SLANT = SEGPATH_SLANT

def get_segpoints_for_n(n: int)-> Sequence[tuple[Sp, Sp]]:
	seg7s: Sequence[Seg7Bit8] = SEG7_DIGIT_ARRAY[n]
	return [SEG7_TO_POINTS[sp] for sp in seg7s]

SEGPOINTS_MAX = len(SEG7_DIGIT_ARRAY)

def get_segpoints_max()-> int:
	return len(SEG7_DIGIT_ARRAY)

def get_all_segpoints_len()-> int:
	return len(SEG7_DIGIT_ARRAY)

def get_all_segpoints()-> list[Sequence[tuple[Sp, Sp]]]:
	lst = []
	for n in range(len(SEG7_DIGIT_ARRAY)):
		lst.append(get_segpoints_for_n(n))
	return lst

def set_all_segpoints(lst: list[Sequence[tuple[Sp, Sp]]]):
	for n in range(len(SEG7_DIGIT_ARRAY)):
		lst[n] = get_segpoints_for_n(n)
from strok7 import SegPath
def get_segpath_for_n(n: int, segpath_list = [None] * len(seg_7_array))-> Sequence[SegPath]:
	for segs in seg_7_array[n]:
		spth_list = []
		for c in segs:
			if c:
				path_pair = get_segpath_for_c(c).path
				spth_list.append(path_pair)
		segpath_list[n] = spth_list
	return segpath_list	

def get_segpath_list(segpath_list = [None] * len(seg_7_array))-> Sequence[Sequence[SegPath]]:
	for i, segs in enumerate(seg_7_array):
		spth_list = []
		for c in segs:
			if c:
				path_pair = get_segpath_for_c(c).path
				spth_list.append(path_pair)
		segpath_list[i] = spth_list
	return segpath_list	

from seg_7_digits import SEG_POINT_PAIR_DIGIT_ARRAY
class BasicDigitStrokes:
	f'''strokes[{SEVEN_SEG_SIZE}]: slanted strokes]'''

	def __init__(self, slant: StrokeSlant=StrokeSlant.SLANT00, scale: int=1, offset: tuple[int, int]=(0, 0)):
		self.slant = slant
		self.scale = scale
		self.offset = offset
		self.get: Callable[[int], list[Sequence[i_i_tpl]]]= lru_cache(maxsize=len(SEG_POINT_PAIR_DIGIT_ARRAY))(self._scale_offset)

	@classmethod
	def get_sp_pairs(cls, n: int)-> Sequence[SpPair]:
		return SEG_POINT_PAIR_DIGIT_ARRAY[n]
	
	def _scale_offset(self, n: int)-> list[Sequence[i_i_tpl]]:
		sp_pairs = self.get_sp_pairs(n)
		return [(spsp.value[0].scale_offset(scale=self.scale, offset=self.offset, slant=self.slant), spsp.value[1].scale_offset(scale=self.scale, offset=self.offset, slant=self.slant)) for spsp in sp_pairs]


class DigitStrokes:
	f'''strokes[{SEVEN_SEG_SIZE}]: slanted strokes]'''
	def __init__(self, cache_factor: int=1): #, max_cache=SEVEN_SEG_SIZE):
		self.strokes: Callable[[int, StrokeSlant, int, tuple[int, int]], Sequence[tuple[i_i_tpl, i_i_tpl]]] = lru_cache(maxsize=SEGPOINTS_MAX * cache_factor)(self._strokes)

	def _strokes(self, n: int, slant: StrokeSlant=StrokeSlant.SLANT00, scale: int=1, offset: tuple[int, int]=(0, 0))-> Sequence[tuple[i_i_tpl, i_i_tpl]]:
		stroke_list = []
		for pp in get_segpoints_for_n(n):
			p_list = [p.scale_offset(slant=slant, scale=scale, offset=offset) for p in pp]
			stroke_list.append(p_list)
		return stroke_list



class NumStrokes(DigitStrokes):
	f'''strokes[{SEVEN_SEG_SIZE}]: slanted strokes]'''
	def __init__(self, scale: float=1.0, offset: tuple[int, int]=(0, 0), slant=NUMSTROKE_SLANT, max_cache=len(homo_seg_7_array)):
		self.scale = scale
		self.offset = offset
		self.slant = slant
		self.strokes: Callable[[int], NDArray] = lru_cache(maxsize=max_cache)(self._strokes)

	def _strokes(self, n: int)-> NDArray:
		stroke_list = [(p1.slant_self(s=self.slant, scale=self.scale, offset=self.offset),
		p2.slant_self(s=self.slant, scale=self.scale, offset=self.offset)) for (p1, p2) in get_segpoints_for_n(n)]
		return stroke_list

if __name__ == '__main__':
	from pprint import pp
	digit_stroke_1 = BasicDigitStrokes(scale=8, offset=(1, 1))
	stroke_1_0 = digit_stroke_1.get(0)
	pp(stroke_1_0)
	digit_stroke_2 = BasicDigitStrokes(scale=16, offset=(2, 2))
	stroke_2_0 = digit_stroke_2.get(0)
	pp(stroke_2_0)
