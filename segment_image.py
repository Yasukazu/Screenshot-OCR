from functools import lru_cache
from collections.abc import Callable
from collections.abc import Sequence
import numpy as np
from PIL import ImageDraw, Image
import numpy as np
from format_num import FormatNum, conv_num_to_bin, formatnums_to_bytearray
from num_strokes import SEGPOINTS_MAX, BasicDigitStrokes
from strok7 import SegElem
from seg_7_digits import hex_to_bit8, str_to_seg_elems # bin2_to_bit8, 
from segment_strokes import SegmentStrokes
from image_fill import ImageFill
from digit_image import DigitImageParam

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
	def calc_scale_from_height(cls, height: int=HEIGHT, padding: int=-1, line_width: int=-1)-> DigitImageParam:
		if height <= 0 or line_width == 0:
			raise ValueError("height or line_width is/are wrong!")
		if padding < 0:
			padding = height // 8
		if line_width < 0:
			line_width = height // 10 or 1
		scale = (height - 2 * padding - line_width) // 2 
		if scale - line_width <= 0:
			raise ValueError(f"scale({scale}) is too small to show!")
		width = scale + 2 * padding + line_width
		return DigitImageParam(width=width, scale=scale, padding=(padding, padding), line_width=line_width)

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

	def __init__(self, param: DigitImageParam, 
			bgcolor=ImageFill.WHITE): # width: int, scale: int, padding: Sequence[int], line_width: int　slant: StrokeSlant=StrokeSlant.SLANT00, 
		stroke_scale = param.scale # - padding[0] * 2 - line_width
		stroke_offset = [pad + param.line_width // 2 or 1 for pad in param.padding]
		self.stroke_feeder = SegmentStrokes(scale=stroke_scale, offset=(stroke_offset[0], stroke_offset[1]))
		self.line_width = param.line_width
		self.bgcolor = bgcolor
		self.size = param.width, param.height
		# self.get: Callable[[int], Image.Image] = lru_cache(maxsize=self.stroke_feeder.get_max())(self._get)

	def get(self, n: int | Sequence[SegElem])-> Image.Image:
		img = Image.new('L', self.size, color=self.bgcolor.value)
		drw = ImageDraw.Draw(img)
		self.stroke_feeder.draw_all(drw=drw, bn=n, line_width=self.line_width, fill=ImageFill.invert(self.bgcolor).value)
		return img
from bin2 import Bin2
def get_number_image(nn: Sequence[int | FormatNum] | bytearray, image_feeder=SegmentImage(param=SegmentImage.calc_scale_from_height(SegmentImage.HEIGHT)), bin2_input=True)-> Image.Image:
	b_array = nn if isinstance(nn, bytearray) else formatnums_to_bytearray(nn, conv_to_bin2=bin2_input)
	number_image_size = len(b_array) * image_feeder.size[0], image_feeder.size[1]
	number_image = Image.new('L', number_image_size, (0,))
	offset = (0, 0)
	x_offset = image_feeder.size[0]
	for n in b_array:
		seg7 = Bin2(n).to_bit8() if bin2_input else hex_to_bit8(n)
		digit_image = image_feeder.get(seg7.value if seg7 else None)
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
	seg_elems = str_to_seg_elems(sys.argv[1])
	hx = float(sys.argv[1])
	nn = [FloatFormatNum(hx, fmt="%.2f")]
	ff2 = [n.conv_to_bin2() for n in nn]
	bb2 = bytearray(*ff2)

#conv_num_to_bin
	bb = formatnums_to_bytearray(nn, conv_to_bin2=True) #conv_num_to_bin(hx, fmt="%x")
	hx_img = get_number_image(bb, image_feeder=s7i, bin2_input=True)
	hx_img.show()
	sys.exit(0)
	for b in bb:
		s7 = hex_to_bit8(b)
		hx_img = s7i.get(s7.value)
		hx_img.show()
