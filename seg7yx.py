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
	def __init__(self, yx: Sequence[Sequence[float]]= [
			[0, 1],
			[0, 1],
			[0, 1],
		]): #scale=1):
		self.yx = yx
		# self.node_list = [scale * Seg7yx.node_dict[x] for x in range(6)]
		# self.scale = scale

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
	def scale_offset(self, scale=1, offset=(0, 0)):
		self._scale(scale)
		self._offset(offset)
	def slant_scale_offset(self, slant=0.0, scale=1, offset=(0, 0)):
		self._slant(slant)
		self._scale(scale)
		self._offset(offset)
	def _offset(self, offset=(0, 0)):
		self.yx = [[xx[0] + offset[0], xx[1] + offset[1]] for xx in (self.yx)]
	def _scale(self, scale=1):
		self.yx = [[xx[0] * scale, xx[1] * scale] for xx in (self.yx)]
	def _slant(self, slant_r: float):
		self.yx = [self.slanted_xx(slant_r, xx, y) for y, xx in enumerate(self.yx)]
	def slanted(self, slant_r: float):
		return [self.slanted_xx(slant_r, xx, y) for y, xx in enumerate(self.yx)]

	@classmethod
	def offset_scaled(cls, offset: Sequence[int], scale: int, xy: Sequence[list[float]]):
		return cls._offset(offset, cls._scale(scale, xy))
	@classmethod	
	def _offset(cls, offset: Sequence[int], xy: Sequence[Sequence[float]]):
		return [[x + offset[0], y + offset[1]] for (x, y) in xy]
	@classmethod
	def _scale(cls, scale: int, xy: Sequence[list[float]]):
		return [[xx[0] * scale, xx[1] * scale] for xx in xy]

	@classmethod
	def slanted_new(cls, slant_r: float, yx: 'Seg7yx')-> 'Seg7yx':
		return cls([cls.slanted_xx(slant_r, xx, y) for y, xx in enumerate(yx.yx)])
	@classmethod
	def slanted_xx(cls, slant: float, xx: Sequence[float], y: int)-> Sequence[float]:
		return [cls.slanted_x(x, y, slant) for x in xx]
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

	s7yx = Seg7yx()
	offset_scaled = Seg7yx.offset_scaled((100,200), 10, s7yx.slanted_seg7(0.2))
	s7yx.slant_scale_offset(0.2, 10, (100, 200))
	pp(s7yx.yx)

	pp(s7yx.to_list())
	s7yx2 = s7yx.slanted(0.2)
	pp(s7yx2)
	pp(Seg7yxSlant.SLANT00)
	pp(Seg7yxSlant.SLANT02)
