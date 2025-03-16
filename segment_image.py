from functools import lru_cache
from collections.abc import Callable
from collections.abc import Sequence
import numpy as np
from PIL import ImageDraw, Image
import numpy as np
from format_num import FormatNum, conv_num_to_bin, formatnums_to_bytearray
from num_strokes import SEGPOINTS_MAX, BasicDigitStrokes
from seg_7_digits import hex_to_seg7
from segment_strokes import SegmentStrokes
from image_fill import ImageFill
from digit_image import BasicDigitImageParam

class SegmentImage:
	WIDTH = 10
	HEIGHT = WIDTH * 2
	SIZE = [WIDTH, HEIGHT]
	PADDING = (2, 2)
	LINE_WIDTH = 2

	@classmethod
	def calc_digit_image_scale(cls, size: int, padding: int, line_width: int)-> int:
		offset = padding + line_width // 2
		scale = size - 2 * offset
		return scale

	@classmethod
	def calc_scale_from_height(cls, height: int=HEIGHT, padding: int=-1, line_width: int=-1)-> BasicDigitImageParam:
		if padding < 0:
			padding = height // 8
		if line_width < 0:
			line_width = height // 10 or 1
		scale = (height - 2 * padding - line_width) // 2 
		if scale - line_width <= 0:
			raise ValueError(f"scale({scale}) is too small to show!")
		width = scale + 2 * padding + line_width
		return BasicDigitImageParam(width=width, scale=scale, padding=(padding, padding), line_width=line_width)

	@classmethod
	def calc_digit_image_scale_from_height(cls, height: int, padding: int, line_width: int)-> int:
		offset = padding + line_width // 2
		scale = height // 2 - 2 * offset
		return scale
	
	@classmethod
	def calc_digit_image_size(cls, scale: int, padding: tuple[int, int]=PADDING, line_width=LINE_WIDTH)-> list[float]:
		stroke_offset = [(pad + line_width // 2 or 1 ) for pad in padding]
		size = scale, scale * 2
		return [sz + 2 * stroke_offset[i] for i, sz in enumerate(size)]

	def __init__(self, param: BasicDigitImageParam, 
			bgcolor=ImageFill.WHITE): # width: int, scale: int, padding: Sequence[int], line_width: int　slant: StrokeSlant=StrokeSlant.SLANT00, 
		stroke_scale = param.scale # - padding[0] * 2 - line_width
		stroke_offset = [pad + param.line_width // 2 or 1 for pad in param.padding]
		self.stroke_feeder = SegmentStrokes(scale=stroke_scale, offset=(stroke_offset[0], stroke_offset[1]))
		self.line_width = param.line_width
		self.bgcolor = bgcolor
		self.size = param.width, param.height
		self.get: Callable[[int], Image.Image] = lru_cache(maxsize=self.stroke_feeder.get_max())(self._get)

	def _get(self, n: int)-> Image.Image:
		img = Image.new('L', self.size, color=self.bgcolor.value)
		drw = ImageDraw.Draw(img)
		self.stroke_feeder.draw_all(drw=drw, bn=n, line_width=self.line_width, fill=ImageFill.invert(self.bgcolor).value)
		return img

def get_hex_array_image(nn: Sequence[int | FormatNum] | bytearray, segment_image_feeder=SegmentImage(param=SegmentImage.calc_scale_from_height(SegmentImage.HEIGHT)), bin2=True)-> Image.Image:
	b_array = nn if isinstance(nn, bytearray) else formatnums_to_bytearray(nn, conv_to_bin2=bin2)
	number_image_size = len(b_array) * segment_image_feeder.size[0], segment_image_feeder.size[1]
	number_image = Image.new('L', number_image_size, (0,))
	offset = (0, 0)
	x_offset = segment_image_feeder.size[0]
	for n in b_array:
		seg7 = hex_to_seg7(n)
		digit_image = segment_image_feeder.get(seg7.value)
		number_image.paste(digit_image, offset)
		offset = offset[0] + x_offset, 0
	return number_image

if __name__ == '__main__':
	import sys
	if len(sys.argv) < 2:
		print("Needs float digit/hex")
		sys.exit(1)
	height = 40
	s7i_prm = SegmentImage.calc_scale_from_height(height)
	s7i = SegmentImage(s7i_prm)
	from format_num import FloatFormatNum, formatnums_to_bytearray
	hx = float(sys.argv[1])
	nn = [FloatFormatNum(hx, fmt="%.1f")]
#conv_num_to_bin
	bb = formatnums_to_bytearray(nn, conv_to_bin2=True) #conv_num_to_bin(hx, fmt="%x")
	hx_img = get_hex_array_image(bb, segment_image_feeder=s7i, bin2=True)
	hx_img.show()
	sys.exit(0)
	for b in bb:
		s7 = hex_to_seg7(b)
		hx_img = s7i.get(s7.value)
		hx_img.show()
