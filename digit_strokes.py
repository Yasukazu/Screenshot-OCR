from typing import Callable, Sequence, Iterator
from functools import lru_cache
from strok7 import StrokeSlant, i_i_tpl, SpPair
from seg_7_digits import SEG_POINT_PAIR_DIGIT_ARRAY, Seg7Bit8, expand_to_sp_pairs, hex_to_bit8
from seg7yx import SlantedNode6
class DigitStrokes:
	f'''strokes[{SEG_POINT_PAIR_DIGIT_ARRAY}]: slanted strokes]'''

	def __init__(self, node6=SlantedNode6.SLANT00, scale: int=1, offset: tuple[int, int]=(0, 0)):
		self.node6 = node6
		self.scale = scale
		self.offset = offset
		self.size = scale, 2 * scale
		self.get: Callable[[int], list[Sequence[tuple[int, int] | None]]]= lru_cache(maxsize=len(SEG_POINT_PAIR_DIGIT_ARRAY))(self._scale_offset)
		self.get_segment: Callable[[int], list[Sequence[tuple[int, int]]]]= lru_cache(maxsize=len(SEG_POINT_PAIR_DIGIT_ARRAY))(self._scale_offset_segment)
		self.sp_pair_extracts: dict[SpPair, tuple[i_i_tpl, i_i_tpl | None]] = {}

	@classmethod
	def get_max(cls)-> int:
		return len(SEG_POINT_PAIR_DIGIT_ARRAY)

	@classmethod
	def get_sp_pairs(cls, n: int)-> Sequence[SpPair]:
		return SEG_POINT_PAIR_DIGIT_ARRAY[n]
	
	def _extract_sp_pair(self, spsp: SpPair)-> tuple[i_i_tpl, i_i_tpl | None]:
		if spsp in self.sp_pair_extracts:
			return self.sp_pair_extracts[spsp]
		extract = spsp.value[0].scale_offset(scale=self.scale, offset=self.offset, node6=self.node6), spsp.value[1].scale_offset(scale=self.scale, offset=self.offset, node6=self.node6) if len(spsp.value) > 1 else None
		self.sp_pair_extracts[spsp] = extract
		return extract
	
	def _scale_offset(self, n: int)-> list[Sequence[tuple[int, int] | None]]:
		sp_pairs = self.get_sp_pairs(n)
		return [(spsp.value[0].scale_offset(scale=self.scale, offset=self.offset, node6=self.node6), spsp.value[1].scale_offset(scale=self.scale, offset=self.offset, node6=self.node6) if len(spsp.value) > 1 else None) for spsp in sp_pairs]
	
	def _scale_offset_segment(self, seg7: Seg7Bit8)-> list[Sequence[tuple[int, int]]]:
		sp_pairs = expand_to_sp_pairs(seg7)
		return [(spsp.value[0].scale_offset(scale=self.scale, offset=self.offset, node6=self.node6), spsp.value[1].scale_offset(scale=self.scale, offset=self.offset, node6=self.node6)) for spsp in sp_pairs]

	@classmethod
	def expand_to_sp_pairs(cls, h: int)-> Sequence[SpPair]:
		seg7 = hex_to_bit8(h)
		return expand_to_sp_pairs(seg7)

if __name__ == '__main__':
	from pprint import pp
	digit_stroke_1 = DigitStrokes(scale=8, offset=(1, 1))
	stroke_1_0 = digit_stroke_1.get(0)
	pp(stroke_1_0)
	digit_stroke_2 = DigitStrokes(scale=16, offset=(2, 2))
	stroke_2_0 = digit_stroke_2.get(0)
	pp(stroke_2_0)
