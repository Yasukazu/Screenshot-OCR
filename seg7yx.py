
type f_list = list[float]
type seg_yx = list[f_list]
class Seg7yx:
	def __init__(self, yx = [
			[0, 1],
			[0, 1],
			[0, 1],
		]):
		self.yx = yx

	def to_list(self): #-> list[list[list[float]]:
		r = []
		for y, xx in enumerate(self.yx):
			r += [[x, y] for x in xx]
		return r

	@classmethod
	def slanted(cls, slant_r: float, yx: 'Seg7yx')-> 'Seg7yx':
		def slant_ratio_by_y(y: int)-> float:
			return 1 - y / 2
		def slanted_x(x: float, y: int, slant: float):
			return x + slant * slant_ratio_by_y(y)
		def slanted_xx(slant: float, xx: list[float], y: int):
			return [slanted_x(x, y, slant) for x in (xx)]
		return cls([slanted_xx(slant_r, xx, y) for y, xx in enumerate(yx.yx)])


if __name__ == '__main__':
	from pprint import pp
	s7yx = Seg7yx()
	pp(s7yx.to_list())
	s7yx2 = Seg7yx.slanted(0.2, s7yx)
	pp(s7yx2.to_list())