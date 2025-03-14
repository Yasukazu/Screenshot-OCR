from typing import Callable, Sequence
from functools import lru_cache
from strok7 import SpPair
from seg_7_digits import SEG_POINT_PAIR_DIGIT_ARRAY, Seg7, expand_to_sp_pairs, hex_to_seg7, expand_to_xy_list_list
import numpy as np
STROKE_SIZE = 17
class SegmentStrokes:
	f'''strokes[{STROKE_SIZE}]'''
	def __init__(self, scale: int=1, offset: tuple[int, int]=(0, 0)): # slant: StrokeSlant=StrokeSlant.SLANT00, 
		# self.slant = slant
		self.scale = scale
		self.offset = offset
		self.offset_ndarray: np.ndarray | None = None
		self.size = scale, 2 * scale
		self.get: Callable[[int], np.ndarray]= lru_cache(maxsize=STROKE_SIZE)(self._scale_offset_array)

	@classmethod
	def get_max(cls)-> int:
		return STROKE_SIZE

	@classmethod
	def get_sp_pairs(cls, n: int)-> Sequence[SpPair]:
		return SEG_POINT_PAIR_DIGIT_ARRAY[n]
	
	def _scale_offset_array(self, n: int)-> np.ndarray:
		array = self.expand_int_to_xy_array(n)
		if not self.offset_ndarray:
			self.offset_ndarray = np.array(self.offset, dtype=np.int16)
		return array * self.scale + self.offset_ndarray


	@classmethod
	def expand_seg7_to_xy_array(cls, seg7: Seg7, _dict: dict[Seg7, np.ndarray] = {})-> np.ndarray:
		if seg7 in _dict:
			return _dict[seg7]
		xy_list_list = expand_to_xy_list_list(seg7)
		array = np.array(xy_list_list)
		_dict[seg7] = array
		return array

	@classmethod
	def expand_int_to_xy_array(cls, n: int, _dict: dict[int, np.ndarray] = {})-> np.ndarray:
		if n in _dict:
			return _dict[n]
		seg7 = hex_to_seg7(n)
		xy_list_list = expand_to_xy_list_list(seg7)
		array = np.array(xy_list_list, dtype=np.int16)
		_dict[n] = array
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
	scale = 8
	offset = (3, 5)
	ss = SegmentStrokes(scale=scale, offset=offset)
	ss_0 = ss.get(0)
	pp(ss_0)
	img_size = (np.array([scale, 2 * scale], dtype=np.int16) + 2 * np.array(offset, dtype=np.int16)).tolist()
	from PIL import Image, ImageDraw
	image = Image.new('L', img_size)
	draw = ImageDraw.Draw(image)
	for stroke in ss_0:
		ss = stroke.tolist()
		draw.line(ss[0] + ss[1], width=1, fill=255)
	image.show()
	xy_array = SegmentStrokes.expand_seg7_to_xy_array(Seg7.A|Seg7.B)
	pp(xy_array)