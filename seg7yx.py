from typing import Sequence
type f_list = list[float]
type seg_yx = list[f_list]
class Seg7yx:
	def __init__(self, yx = (
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
	@classmethod
	def slanted(cls, slant_r: float, yx: 'Seg7yx')-> 'Seg7yx':
		return cls(tuple([cls.slanted_xx(slant_r, xx, y) for y, xx in enumerate(yx.yx)]))
	@classmethod
	def slanted_xx(cls, slant: float, xx: list[float], y: int)-> Sequence[float]:
		return tuple([cls.slanted_x(x, y, slant) for x in (xx)])
	@classmethod
	def slanted_x(cls, x: float, y: int, slant: float)-> float:
		return x + slant * cls.slant_ratio_by_y(y)
	@classmethod
	def slant_ratio_by_y(cls, y: int)-> float:
		return 1 - y / 2


if __name__ == '__main__':
	from pprint import pp
	s7yx = Seg7yx()
	pp(s7yx.to_list())
	s7yx2 = Seg7yx.slanted(0.2, s7yx)
	pp(s7yx2.to_list())