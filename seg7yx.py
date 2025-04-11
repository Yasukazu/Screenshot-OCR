from typing import Sequence, Iterator
from enum import Enum
from dataclasses import dataclass
type _f_f_ = tuple[float, float]
#type node6 = tuple[_f_f_, _f_f_, _f_f_, _f_f_, _f_f_, _f_f_]

@dataclass
class Node6:
	_nodes: Sequence[_f_f_]
	def __post_init__(self):
		if len(self._nodes) != 6:
			raise ValueError(f"Sequence length({len(self._nodes)=}) is not 6!")
class Seg7yx:
	'''3-row 2-column float array'''
	node_dict = {
		0: (0, 0), 1: (1, 0),
		5: (0, 1), 2: (1, 1),
		4: (0, 2), 3: (1, 2)
	}
	def __init__(self, yx: Sequence[Sequence[float]]= [
			(0, 1),
			(0, 1),
			(0, 1),
		], slant=0.0):
		self.yx = yx
		self.slant = slant
		#if slant > 0:			self._slant(slant)
		# self.node_list = [scale * Seg7yx.node_dict[x] for x in range(6)]
		# self.scale = scale

	def _to_list(self)-> list[tuple[float, float]]:
		r = []
		for y, xx in enumerate(self.yx):
			r.append(tuple([(x, y) for x in xx]))
		return (r)
	
	def to_node6(self):
		'''dst[2] = src[3]
		dst[3] = src[5]
		dst[5] = src[2]'''
		src = self._to_list()
		return src[0], src[1], src[3], src[5], src[4], src[2]

	def slanted_node6(self):#, slant_r=0.0):#-> Node6:
		slanted = [self.slanted_xx(slant=self.slant, xx=xx, y=y) for y, xx in enumerate(self.yx)]
		src = []
		for y, xx in enumerate(slanted):
			src.append(tuple([(x, y) for x in xx]))
		dst = src.copy()
		dst[2] = src[3]
		dst[3] = src[5]
		dst[5] = src[2]
		return tuple(dst)

	def to_tuple(self)-> Sequence[Sequence[float]]:
		r = []
		for y, xx in enumerate(self.yx):
			r += [(x, y) for x in xx]
		return tuple(r)

	#def _slant(self, slant_r: float):		self.yx = [self.slanted_xx(slant_r, xx, y) for y, xx in enumerate(self.yx)]

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
type seq_f_f = Sequence[tuple[float, float]]
class Seg7Node6:
	def __init__(self, slant=0.0):
		self.slant = slant
	@property
	def node6(self):#, slant_r=0.0):
		return Seg7yx(slant=self.slant).slanted_node6()
	@classmethod
	def scale_offset(cls, scale: int, offset: Sequence[int], nodes: seq_f_f):
		return cls._offset(offset, cls._scale(scale, nodes))
	@classmethod
	def _offset(cls, offset: Sequence[int], nodes: seq_f_f):
		return [(x + offset[0], y + offset[1]) for (x, y) in nodes]
	@classmethod
	def _scale(cls, scale: int, nodes: seq_f_f):
		return [(xx[0] * scale, xx[1] * scale) for xx in nodes]

class SlantedNode6(Enum):
	SLANT00 = Seg7yx().to_node6() # [[0, 0], [1, 0], [1, 1], [1, 2], [0, 2], [0, 1]]
	SLANT02 = Seg7yx(slant=0.2).slanted_node6() # [[0.2, 0], [1.2, 0], [1.1, 1], [1.0, 2], [0.0, 2], [0.1, 1]]

if __name__ == '__main__':
	from pprint import pp
	# s7yx = Seg7yx() s7yx.slanted_seg7
	seg7node6 = Seg7Node6(slant=0.2)
	scale10_offset100x200_seg7node6 = Seg7Node6.scale_offset(10, (100, 200), seg7node6.node6)
	scale20_offset200x300_seg7node6 = Seg7Node6.scale_offset(20, (200, 300), seg7node6.node6)
	print()
	#offset_scaled = Seg7yx.offset_scaled((100,200), 10, s7yx.slanted_seg7(0.2))
	#s7yx.slant_scale_offset(0.2, 10, (100, 200))
	#pp(s7yx.yx)
	#pp(s7yx.to_list())
	#s7yx2 = s7yx.slanted(0.2)
	#pp(s7yx2)
	#pp(Seg7yxSlant.SLANT00)
	#pp(Seg7yxSlant.SLANT02)
