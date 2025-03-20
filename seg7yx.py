from typing import Sequence
type f_list = list[float]
type seg_yx = list[f_list]
class Seg7yx:
	def __init__(self, yx: Sequence[Sequence[float]]= (
			(0, 1),
			(0, 1),
			(0, 1),
		)):
		self.yx = yx

	def to_list(self)-> list[list[float]]:
		r = []
		for y, xx in enumerate(self.yx):
			r += [[x, y] for x in xx]
		return r
	def to_tuple(self)-> Sequence[Sequence[float]]:
		r = []
		for y, xx in enumerate(self.yx):
			r += [(x, y) for x in xx]
		return tuple(r)
	
	def expand_y(self):
		r = []
		for i, xx in enumerate(self.yx):
			r += [(x, i) for x in xx]
		return tuple(r)

	def slanted_yx(self, slant_r: float)-> Sequence[Sequence[float]]:
		return tuple([self.slanted_xx(slant_r, xx, y) for y, xx in enumerate(self.yx)])
	@classmethod
	def slanted_new(cls, slant_r: float, yx: 'Seg7yx')-> 'Seg7yx':
		return cls(tuple([cls.slanted_xx(slant_r, xx, y) for y, xx in enumerate(yx.yx)]))
	@classmethod
	def slanted_xx(cls, slant: float, xx: Sequence[float], y: int)-> Sequence[float]:
		return tuple([cls.slanted_x(x, y, slant) for x in xx])
	@classmethod
	def slanted_x(cls, x: float, y: int, slant: float)-> float:
		return x + slant * cls.slant_ratio_by_y(y)
	@classmethod
	def slant_ratio_by_y(cls, y: int)-> float:
		return 1 - y / 2

from enum import Enum
class Seg7yxSlant(Enum):
	SLANT00 = Seg7yx().yx # ((0, 1), (0, 1), (0, 1))
	SLANT02 = Seg7yx().slanted_yx(0.2) # ((0.2, 1.2), (0.1, 1.1), (0.0, 1.0))
class SlantedSeg7yx(Enum):
	SLANT00 = Seg7yx() # ((0, 1), (0, 1), (0, 1))
	SLANT02 = Seg7yx().slanted_new(0.2, SLANT00) # ((0.2, 1.2), (0.1, 1.1), (0.0, 1.0))
if __name__ == '__main__':
	from pprint import pp
	sl0 = SlantedSeg7yx.SLANT00
	sl2 = SlantedSeg7yx.SLANT02
	seg_points = sl2.value.expand_y()
	import numpy as np
	offset = np.array([55, 77])
	scale_offset_seg_points = 7 * np.array(seg_points) + offset
	s7yx = Seg7yx()
	pp(s7yx.to_list())
	s7yxn = Seg7yx.slanted_new(0.2, s7yx)
	s7yx2 = s7yx.slanted_yx(0.2)
	pp(s7yx2)