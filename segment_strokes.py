from typing import Callable, Sequence, Iterator
from functools import lru_cache
from strok7 import StrokeSlant, i_i_tpl, SpPair
from seg_7_digits import SEG_POINT_PAIR_DIGIT_ARRAY, Seg7, expand_to_sp_pairs, hex_to_seg7, expand_to_xy_list_list
import numpy as np

class SegmentStrokes:
	f'''strokes[{SEG_POINT_PAIR_DIGIT_ARRAY}]: slanted strokes]'''

	def __init__(self, slant: StrokeSlant=StrokeSlant.SLANT00, scale: int=1, offset: tuple[int, int]=(0, 0)):
		self.slant = slant
		self.scale = scale
		self.offset = offset
		self.size = scale, 2 * scale
		self.get: Callable[[int], list[Sequence[tuple[int, int]]]]= lru_cache(maxsize=len(SEG_POINT_PAIR_DIGIT_ARRAY))(self._scale_offset)
		self.get_segment: Callable[[int], list[Sequence[tuple[int, int]]]]= lru_cache(maxsize=len(SEG_POINT_PAIR_DIGIT_ARRAY))(self._scale_offset_segment)
		self._seg7_to_xy_ndarray_dict: dict[Seg7, np.ndarray] = {}
		self.sp_pair_extracts: dict[SpPair, tuple[i_i_tpl, i_i_tpl]] = {}

	@classmethod
	def get_max(cls)-> int:
		return len(SEG_POINT_PAIR_DIGIT_ARRAY)

	@classmethod
	def get_sp_pairs(cls, n: int)-> Sequence[SpPair]:
		return SEG_POINT_PAIR_DIGIT_ARRAY[n]
	
	def _extract_sp_pair(self, spsp: SpPair)-> tuple[i_i_tpl, i_i_tpl]:
		if spsp in self.sp_pair_extracts:
			return self.sp_pair_extracts[spsp]
		extract = spsp.value[0].scale_offset(scale=self.scale, offset=self.offset, slant=self.slant), spsp.value[1].scale_offset(scale=self.scale, offset=self.offset, slant=self.slant)
		self.sp_pair_extracts[spsp] = extract
		return extract
	
	def _scale_offset(self, n: int)-> list[Sequence[tuple[int, int]]]:
		sp_pairs = self.get_sp_pairs(n)
		return [(spsp.value[0].scale_offset(scale=self.scale, offset=self.offset, slant=self.slant), spsp.value[1].scale_offset(scale=self.scale, offset=self.offset, slant=self.slant)) for spsp in sp_pairs]
	
	def _scale_offset_segment(self, seg7: Seg7)-> list[Sequence[tuple[int, int]]]:
		sp_pairs = expand_to_sp_pairs(seg7)
		return [(spsp.value[0].scale_offset(scale=self.scale, offset=self.offset, slant=self.slant), spsp.value[1].scale_offset(scale=self.scale, offset=self.offset, slant=self.slant)) for spsp in sp_pairs]

	@classmethod
	def expand_seg7_to_xy_ndarray(cls, seg7: Seg7, _dict: dict[Seg7, np.ndarray] = {})-> np.ndarray:
		if seg7 in _dict:
			return _dict[seg7]
		xy_list_list = expand_to_xy_list_list(seg7)
		array = np.array(xy_list_list)
		_dict[seg7] = array
		return array


	@classmethod
	def expand_to_segments(cls, h: int)-> Sequence[Seg7]:
		seg7 = hex_to_seg7(h)
		elements = []
		for seg in [Seg7.A, Seg7.B, Seg7.C, Seg7.D, Seg7.E, Seg7.F, Seg7.G,]:
			if seg7 & seg:
				elements.append(seg)
		return tuple(elements)

if __name__ == '__main__':
	from pprint import pp
	digit_stroke_1 = SegmentStrokes(scale=8, offset=(1, 1))
	stroke_1_0 = digit_stroke_1.get(0)
	pp(stroke_1_0)
	digit_stroke_2 = SegmentStrokes(scale=16, offset=(2, 2))
	stroke_2_0 = digit_stroke_2.get(0)
	pp(stroke_2_0)
