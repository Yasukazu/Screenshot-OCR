from typing import Callable, Sequence
from functools import lru_cache
from strok7 import StrokeSlant, i_i_tpl, SpPair
from seg_7_digits import SEG_POINT_PAIR_DIGIT_ARRAY

class DigitStrokes:
	f'''strokes[{SEG_POINT_PAIR_DIGIT_ARRAY}]: slanted strokes]'''

	def __init__(self, slant: StrokeSlant=StrokeSlant.SLANT00, scale: int=1, offset: tuple[int, int]=(0, 0)):
		self.slant = slant
		self.scale = scale
		self.offset = offset
		self.size = scale, 2 * scale
		self.get: Callable[[int], list[Sequence[tuple[int, int]]]]= lru_cache(maxsize=len(SEG_POINT_PAIR_DIGIT_ARRAY))(self._scale_offset)

	@classmethod
	def get_max(cls)-> int:
		return len(SEG_POINT_PAIR_DIGIT_ARRAY)

	@classmethod
	def get_sp_pairs(cls, n: int)-> Sequence[SpPair]:
		return SEG_POINT_PAIR_DIGIT_ARRAY[n]
	
	def _scale_offset(self, n: int)-> list[Sequence[tuple[int, int]]]:
		sp_pairs = self.get_sp_pairs(n)
		return [(spsp.value[0].scale_offset(scale=self.scale, offset=self.offset, slant=self.slant), spsp.value[1].scale_offset(scale=self.scale, offset=self.offset, slant=self.slant)) for spsp in sp_pairs]

if __name__ == '__main__':
	from pprint import pp
	digit_stroke_1 = DigitStrokes(scale=8, offset=(1, 1))
	stroke_1_0 = digit_stroke_1.get(0)
	pp(stroke_1_0)
	digit_stroke_2 = DigitStrokes(scale=16, offset=(2, 2))
	stroke_2_0 = digit_stroke_2.get(0)
	pp(stroke_2_0)
