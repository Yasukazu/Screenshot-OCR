from typing import Callable, Sequence
from types import MappingProxyType
from functools import lru_cache
from numpy.typing import NDArray
from strok7 import SEG_POINTS, SEGPATH_SLANT, get_segpath_for_c, Sp0, StrokeSlant, f_i_tpl
from seg_7_digits import Seg7, SEG7_ARRAY, seg_7_array, homo_seg_7_array
from seven_seg import SEVEN_SEG_SIZE

SEG7_TO_POINTS: dict[Seg7, tuple[Sp0, Sp0]] = MappingProxyType({
	Seg7.A: (SEG_POINTS[0], SEG_POINTS[1]),
	Seg7.B: (SEG_POINTS[1], SEG_POINTS[2]),
	Seg7.C: (SEG_POINTS[2], SEG_POINTS[3]),
	Seg7.D: (SEG_POINTS[3], SEG_POINTS[4]),
	Seg7.E: (SEG_POINTS[4], SEG_POINTS[5]),
	Seg7.F: (SEG_POINTS[5], SEG_POINTS[0]),
	Seg7.G: (SEG_POINTS[5], SEG_POINTS[2]),
})

NUMSTROKE_SLANT = SEGPATH_SLANT

def get_segpoints_for_n(n: int)-> Sequence[tuple[Sp0, Sp0]]:
	seg7s: Sequence[Seg7] = SEG7_ARRAY[n]
	return (SEG7_TO_POINTS[sp] for sp in seg7s)

SEGPOINTS_MAX = len(SEG7_ARRAY)

def get_segpoints_max()-> int:
	return len(SEG7_ARRAY)

def get_all_segpoints()-> list[Sequence[tuple[Sp0, Sp0]]]:
	lst = []
	for n in range(len(SEG7_ARRAY)):
		lst.append(get_segpoints_for_n(n))
	return lst

def get_segpath_for_n(n: int, segpath_list = [None] * len(seg_7_array))-> Sequence[tuple[Sp0, Sp0]]:
	for segs in seg_7_array[n]:
		spth_list = []
		for c in segs:
			if c:
				path_pair: tuple[Sp0,Sp0] = get_segpath_for_c(c).path
				spth_list.append(path_pair)
		segpath_list[n] = spth_list
	return segpath_list	

def get_segpath_list(segpath_list = [None] * len(seg_7_array))-> Sequence[Sequence[tuple[Sp0, Sp0]]]:
	for i, segs in enumerate(seg_7_array):
		spth_list = []
		for c in segs:
			if c:
				path_pair: tuple[Sp0,Sp0] = get_segpath_for_c(c).path
				spth_list.append(path_pair)
		segpath_list[i] = spth_list
	return segpath_list	

class DigitStrokes:
	f'''strokes[{SEVEN_SEG_SIZE}]: slanted strokes]'''
	#_segpath_list: list[tuple[Sp0, Sp0]] = get_segpath_list()
	def __init__(self, cache_factor: int=1): #, max_cache=SEVEN_SEG_SIZE):
		self.strokes: Callable[[int], list[tuple[tuple[int, int]]]] = lru_cache(maxsize=SEGPOINTS_MAX * cache_factor)(self._strokes)

	def _strokes(self, n: int, slant: StrokeSlant=StrokeSlant.SLANT02, scale: int=1, offset: tuple[int, int]=(0, 0))-> Sequence[tuple[i_i_tpl, i_i_tpl]]:
		stroke_list = []
		for pp in get_segpoints_for_n(n): # self._segpath_list[n]:
			p_list = [p.scale_offset(slant=slant, scale=scale, offset=offset) for p in pp] # p_list = ((scale * p.slant(slant) + offset[i]) for i, p in enumerate(pp))
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
		stroke_list = [(p1.slant(s=self.slant, scale=self.scale, offset=self.offset),
		p2.slant(s=self.slant, scale=self.scale, offset=self.offset)) for (p1, p2) in get_segpoints_for_n(n)]
		return stroke_list

if __name__ == '__main__':
	from pprint import pp
	import cProfile
	offset=(7,3)
	scale=2

	all_strokes_list = get_all_segpoints()
	num_strokes = DigitStrokes()
	segpoints_range = get_segpoints_max()
	strokes_list = [
	num_strokes.strokes(n) for n in range(segpoints_range)
	]
	pp(strokes_list)
	strokes_0 = num_strokes.strokes(0)
	pp(strokes_0)
	fcall = f"num_strokes.strokes(0, scale={scale}, offset={offset})"
	cProfile.run(fcall)
	print('2nd:')
	cProfile.run(fcall)


