from enum import Enum
from collections import namedtuple
from functools import lru_cache
from collections.abc import Callable
from collections.abc import Sequence
import numpy as np
from PIL import ImageDraw, Image
import numpy as np
from strok7 import STRK_DICT_STEM, StrokeSlant, i_i_tpl
from format_num import FormatNum, HexFormatNum, conv_num_to_bin, formatnums_to_bytearray
from num_strokes import SEGPOINTS_MAX, DigitStrokes, BasicDigitStrokes
from seg_7_digits import hex_to_seg7
from segment_strokes import SegmentStrokes

class ImageFill(Enum): # single element tuple for ImageDraw color
	BLACK = (0,)
	WHITE = (0xff,)
	@classmethod
	def invert(cls, fill):
		if fill == ImageFill.BLACK:
			return ImageFill.WHITE
		return ImageFill.BLACK

OfstIWIH = namedtuple('OfstIWIH', ['ofst', 'img_w', 'img_h'])
DigitImageCalcResult = namedtuple('DigitImageCalcResult', ['scale', 'font_scale', 'padding', 'line_width'])
class DigitImage:

	MIN_SCALE = 20
	STANDARD_PADDING = 1
	STANDARD_PADDING_RATIO = 0.25
	STANDARD_LINE_WIDTH = 2
	STANDARD_LINE_WIDTH_RATIO = 0.2

	def __init__(self, stroke_feeder=BasicDigitStrokes(), bgcolor=ImageFill.WHITE, line_width=STANDARD_LINE_WIDTH): # slant: StrokeSlant=StrokeSlant.SLANT00, scale: int=MIN_SCALE,  offset=(STANDARD_PADDING, STANDARD_PADDING)#
		self.stroke_feeder = stroke_feeder
		self.slant = stroke_feeder.slant
		self.scale = stroke_feeder.scale
		self.offset = stroke_feeder.offset
		self.line_width = line_width
		self.bgcolor = bgcolor
		self.get: Callable[[int], Image.Image] = lru_cache(maxsize=SEGPOINTS_MAX)(self._get)

	@classmethod
	def calc_font_scale(cls, scale: int=MIN_SCALE, line_width_ratio: float=STANDARD_LINE_WIDTH_RATIO, padding_ratio: float=STANDARD_PADDING_RATIO)-> DigitImageCalcResult:
		padding = int(scale * padding_ratio)
		f_scale = scale - 2 * padding
		line_width = int(f_scale * line_width_ratio) or 1
		if f_scale - line_width <= 2:
			raise ValueError(f"Insufficient scale:{scale} for line-width-ratio{line_width_ratio}!")	
		return DigitImageCalcResult(scale=scale, font_scale=f_scale, padding=padding, line_width=line_width)

	def _get(self, n: int)-> Image.Image:
		img_w = self.stroke_feeder.scale + 2 * self.stroke_feeder.offset[0]
		img_h = 2 * self.stroke_feeder.scale + 2 * self.stroke_feeder.offset[1]
		img = Image.new('L', (img_w, img_h), color=self.bgcolor.value)
		drw = ImageDraw.Draw(img)
		strokes = self.stroke_feeder.get(n)
		for stroke in strokes:
			drw.line(stroke, width=self.line_width, fill=ImageFill.invert(self.bgcolor).value, joint='curve')
		return img


	def calc_scale_padding(self, scale=MIN_SCALE, padding=STANDARD_PADDING_RATIO, slant: StrokeSlant=StrokeSlant.SLANT02)-> tuple[int, float]:
		padding = scale * padding or 1
		_scale = scale - 2 * padding #  int(scale * (1 - slant.value))
		return int(_scale * (1 - slant.value) * padding or 4 * padding), padding

import digit_strokes
from collections import namedtuple
from dataclasses import dataclass, field
@dataclass
class BasicDigitImageParam:
	width: int
	scale: int
	padding: tuple[int, int]
	line_width: int
	height: int = field(init=False)
	def __post_init__(self):
		if min(self.padding) < 0:
			raise ValueError("Every padding value must be larger than 0!")
		if self.line_width < 0:
			raise ValueError("The value of line_width must be larger than 0!")
		if self.scale - self.line_width <= 0:
			raise ValueError("The value of scale must be larger than line_width!")
		self.height = self.width + self.scale

class Seg7Image:
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
		strokes = self.stroke_feeder.get(n)
		for stroke in strokes:
			drw.line(stroke.ravel().tolist(), width=self.line_width, fill=ImageFill.invert(self.bgcolor).value, joint='curve')
		return img

def get_hex_array_image(nn: Sequence[int | FormatNum] | bytearray, seg7_image_feeder=Seg7Image(param=Seg7Image.calc_scale_from_height(Seg7Image.HEIGHT)))-> Image.Image:
	b_array = nn if isinstance(nn, bytearray) else formatnums_to_bytearray(nn)
	number_image_size = len(b_array) * seg7_image_feeder.size[0], seg7_image_feeder.size[1]
	number_image = Image.new('L', number_image_size, (0,))
	offset = (0, 0)
	x_offset = seg7_image_feeder.size[0]
	for n in b_array:
		seg7 = hex_to_seg7(n)
		digit_image = seg7_image_feeder.get(seg7.value)
		number_image.paste(digit_image, offset)
		offset = offset[0] + x_offset, 0
	return number_image

if __name__ == '__main__':
	import sys
	if len(sys.argv) < 2:
		print("Needs digit/hex")
		sys.exit(1)
	height = 40
	s7i_prm = Seg7Image.calc_scale_from_height(height)
	s7i = Seg7Image(s7i_prm)
	hx = int(sys.argv[1], 16)
	from format_num import conv_num_to_bin
	bb = conv_num_to_bin(hx, fmt="%x")
	hx_img = get_hex_array_image(bb)
	hx_img.show()
	sys.exit(0)
	for b in bb:
		s7 = hex_to_seg7(b)
		hx_img = s7i.get(s7.value)
		hx_img.show()
