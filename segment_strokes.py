from typing import Callable, Sequence
from functools import lru_cache
from PIL import ImageDraw
from strok7 import SpPair
from seg_7_digits import SEG_POINT_PAIR_DIGIT_ARRAY, Bit8, expand_to_sp_pairs, hex_to_seg7, expand_to_xy_list_list, bin_to_seg7, expand_bin_to_xy_list_list, expand_bin_to_seg_elems

import numpy as np
STROKE_SIZE = 18
class SegmentStrokes:
	f'''strokes[{STROKE_SIZE}]: Use hex_to_seg7 to get from digit'''

	def __init__(self, scale: int=1, offset: tuple[int, int]=(0, 0)): # slant: StrokeSlant=StrokeSlant.SLANT00, 
		# self.slant = slant
		self.scale = scale
		self.offset = offset
		self.offset_ndarray: np.ndarray | None = None
		self.size = scale, 2 * scale
		self.get: Callable[[int], np.ndarray]= lru_cache(maxsize=STROKE_SIZE)(self._scale_offset_bin)

	@classmethod
	def get_max(cls)-> int:
		return STROKE_SIZE

	@classmethod
	def get_sp_pairs(cls, n: int)-> Sequence[SpPair]:
		return SEG_POINT_PAIR_DIGIT_ARRAY[n]
	
	def _scale_offset_array(self, n: int)-> np.ndarray:
		array = self.expand_int_to_xy_array(n)
		if not self.offset_ndarray:
			self.offset_ndarray = np.array(self.offset, dtype=np.int64)
		return array * self.scale + self.offset_ndarray

	def _scale_offset_bin(self, bn: int)-> np.ndarray:
		array = self._expand_bin_to_xy_array(bn)
		if self.offset_ndarray is None:
			self.offset_ndarray = np.array(self.offset, dtype=np.int64)
		return array * self.scale + self.offset_ndarray
	
	def draw(self, drw: ImageDraw.ImageDraw, bn: int, line_width=1, fill=0):
		seg_elems = expand_bin_to_seg_elems(bn)
		for seg_elem in seg_elems:
			seg_path = seg_elem.value
			seg_path.draw(drw=drw, scale=self.scale, offset=self.offset, line_width=line_width, fill=fill)

	def scale_offset(self, seg: Bit8, _dict: dict[Bit8, np.ndarray] = {})-> np.ndarray:
		if seg in _dict:
			return _dict[seg]
		array = self.expand_seg7_to_xy_array(seg)
		if not self.offset_ndarray:
			self.offset_ndarray = np.array(self.offset, dtype=np.int64)
		value = array * self.scale + self.offset_ndarray
		_dict[seg] = value
		return value

	@classmethod
	def _expand_bin_to_xy_array(cls, bn: int)-> np.ndarray:
		return np.array(expand_bin_to_xy_list_list(bn), dtype=np.int64)

	@classmethod
	def expand_seg7_to_xy_array(cls, seg7: Bit8, _dict: dict[Bit8, np.ndarray] = {})-> np.ndarray:
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
		array = np.array(xy_list_list, dtype=np.int64)
		_dict[n] = array
		return array

	@classmethod
	def expand_to_segments(cls, h: int)-> Sequence[Bit8]:
		seg7 = hex_to_seg7(h)
		elements = []
		for seg in [Bit8.A, Bit8.B, Bit8.C, Bit8.D, Bit8.E, Bit8.F, Bit8.G,]:
			if seg7 & seg:
				elements.append(seg)
		return tuple(elements)

if __name__ == '__main__':
	from pprint import pp
	from PIL import Image, ImageDraw
	from format_num import FloatFormatNum
	scale = 80
	offset = (10, 20)
	ss = SegmentStrokes(scale=scale, offset=offset)
	img_size = (np.array([scale, 2 * scale], dtype=np.int64) + 2 * np.array(offset, dtype=np.int64)).tolist()
	fmtnum = FloatFormatNum(0.1, fmt="%.1f")
	n_bb = fmtnum.conv_to_bin()
	for n_b in n_bb:
		n_seg = hex_to_seg7(n_b)
		n_bin = n_seg.value
		image = Image.new('L', img_size, 0xff)
		draw = ImageDraw.Draw(image)
		ss.draw(drw=draw, bn=n_bin, line_width=4)
		image.show()
#	seg = Bit8.H	a_bin = seg.value
	'''
	abcd_bin = abcd_seg.value
	n = 0
	n_seg7 = hex_to_seg7(n)
	n_bin = n_seg7.value
	abcd_strokes = ss.get(n_bin)
	pp(abcd_strokes)
		abcd_seg = Seg7.A | Seg7.B | Seg7.C | Seg7.D
	abcd_bin = abcd_seg.value
		seg_0 = hex_to_seg7(0)
	seg_0_bin = seg_0.value
	strokes = ss.get(seg_0_bin) #_scale_offset_bin

	for stroke in strokes:
		ss = stroke.tolist()
		draw.line(ss[0] + ss[1], width=1, fill=255)
	image.show()
	xy_array = SegmentStrokes.expand_seg7_to_xy_array(Seg7.A|Seg7.B)
	pp(xy_array)
'''