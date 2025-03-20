from typing import Sequence
type f_list = list[float]
type seg_yx = list[f_list]
class Seg7yx:
	'''3-row 2-column float array'''
	node_dict = {
		0: (0, 0), 1: (1, 0),
		5: (0, 1), 2: (1, 1),
		4: (0, 2), 3: (1, 2)
	}
	def __init__(self, yx: Sequence[Sequence[float]]= (
			(0, 1),
			(0, 1),
			(0, 1),
		), scale=1):
		self.yx = yx
		# self.node_list = [scale * Seg7yx.node_dict[x] for x in range(6)]
		self.scale = scale

	def to_list(self)-> list[list[float]]:
		r = []
		for y, xx in enumerate(self.yx):
			r += [[x, y] for x in xx]
		return r
	def to_seg7(self)-> list[list[float]]:
		src = self.to_list()
		dst = src.copy()
		dst[2] = src[3]
		dst[3] = src[5]
		dst[5] = src[2]
		return dst
	def slanted_seg7(self, slant_r: float):
		slanted = [self.slanted_xx(slant_r, xx, y) for y, xx in enumerate(self.yx)]
		src = []
		for y, xx in enumerate(slanted):
			src += [[x, y] for x in xx]
		dst = src.copy()
		dst[2] = src[3]
		dst[3] = src[5]
		dst[5] = src[2]
		return dst
	def to_tuple(self)-> Sequence[Sequence[float]]:
		r = []
		for y, xx in enumerate(self.yx):
			r += [(x, y) for x in xx]
		return tuple(r)
	def slanted(self, slant_r: float):
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
	SLANT00 = Seg7yx().to_seg7() # [[0, 0], [1, 0], [1, 1], [1, 2], [0, 2], [0, 1]]
	SLANT02 = Seg7yx().slanted_seg7(0.2) # [[0.2, 0], [1.2, 0], [1.1, 1], [1.0, 2], [0.0, 2], [0.1, 1]]

if __name__ == '__main__':
	from pprint import pp
	pp(Seg7yxSlant.SLANT00)
	pp(Seg7yxSlant.SLANT02)
	s7yx = Seg7yx()
	pp(s7yx.to_list())
	s7yx2 = s7yx.slanted(0.2)
	pp(s7yx2)