from enum import Enum
from collections import namedtuple
from functools import lru_cache
from collections.abc import Callable
from collections.abc import Sequence
from PIL import ImageDraw, Image
import numpy as np
from strok7 import STRK_DICT_STEM, StrokeSlant, i_i_tpl
from format_num import FormatNum, HexFormatNum, conv_num_to_bin, formatnums_to_bytearray
from num_strokes import SEGPOINTS_MAX, DigitStrokes, BasicDigitStrokes
from segment_strokes import hex_to_bit8, SegmentStrokes
from seg_node_pair import DigitStrokeFeeder, encode_str_to_seg7bit8
from seg7bit8 import Seg7Bit8

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

	def __init__(self, stroke_feeder: DigitStrokeFeeder, bgcolor=ImageFill.WHITE, line_width=STANDARD_LINE_WIDTH): # slant: StrokeSlant=StrokeSlant.SLANT00, scale: int=MIN_SCALE,  offset=(STANDARD_PADDING, STANDARD_PADDING)#=BasicDigitStrokes()
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

	WIDTH = 10
	HEIGHT = WIDTH * 2
	SIZE = [WIDTH, HEIGHT]
	PADDING = (2, 2)
	LINE_WIDTH = 2

	def __post_init__(self):
		if min(self.padding) < 0:
			raise ValueError("Every padding value must be larger than 0!")
		if self.line_width < 0:
			raise ValueError("The value of line_width must be larger than 0!")
		if self.scale - self.line_width <= 0:
			raise ValueError("The value of scale must be larger than line_width!")
		self.height = self.width + self.scale

	@classmethod
	def calc_scale_from_height(cls, height: int=HEIGHT, padding: int=-1, line_width: int=-1)-> 'BasicDigitImageParam':
		if padding < 0:
			padding = height // 8
		if line_width < 0:
			line_width = height // 10 or 1
		scale = (height - 2 * padding - line_width) // 2 
		if scale - line_width <= 0:
			raise ValueError(f"scale({scale}) is too small to show!")
		width = scale + 2 * padding + line_width
		return BasicDigitImageParam(width=width, scale=scale, padding=(padding, padding), line_width=line_width)

class DigitDraw:

	def __init__(self, stroke_feeder: DigitStrokeFeeder, param: BasicDigitImageParam,
			bgcolor=ImageFill.WHITE):
		self.stroke_feeder = stroke_feeder
		self.param = param
		self.bgcolor = bgcolor
		self.get: Callable[[Seg7Bit8], Image.Image] = lru_cache(maxsize=self.stroke_feeder.max_size)(self._get)
	
	@property
	def line_width(self):
		return self.param.line_width
	
	@property
	def size(self):
		return self.param.width, self.param.height
	
	def _get(self, s: Seg7Bit8)-> Image.Image:
		img = Image.new('L', self.size, color=self.bgcolor.value)
		drw = ImageDraw.Draw(img)
		digit_strokes = self.stroke_feeder.feed_digit(s)
		for strokes in digit_strokes:
			for stroke in strokes:
				drw.line(stroke, width=self.line_width, fill=ImageFill.invert(self.bgcolor).value, joint='curve')
		return img

def get_basic_number_image(nn: Sequence[int | FormatNum] | bytearray, digit_image_feeder: DigitDraw)-> Image.Image: #BasicDigitImage.WIDTH, (param=BasicDigitImage.calc_scale_from_height(BasicDigitImage.HEIGHT))
	from seg7bit8 import BIN1_TO_SEG7BIT8, bin2_to_seg7bit8
	from bin2 import Bin2
	b_array = nn if isinstance(nn, bytearray) else formatnums_to_bytearray(nn)
	number_image_size = len(b_array) * digit_image_feeder.size[0], digit_image_feeder.size[1]
	number_image = Image.new('L', number_image_size, (0,))
	offset = (0, 0)
	x_offset = digit_image_feeder.size[0]
	for b in b_array:
		b2 = Bin2(b)
		s7b8 = bin2_to_seg7bit8(b2)
		digit_image = digit_image_feeder.get(s7b8)
		number_image.paste(digit_image, offset)
		offset = offset[0] + x_offset, 0
	return number_image

if __name__ == '__main__':
	import sys
	from pprint import pp
	height = 40
	bdiprm = DigitDraw.calc_scale_from_height(height)
	width=bdiprm.width
	scale=bdiprm.scale
	padding=bdiprm.padding
	line_width=bdiprm.line_width
	stroke_feeder = DigitStrokeFeeder(scale=scale, offset=padding)
	num = 3.12
	num_str = "%.2f" % num
	digits = list(encode_str_to_seg7bit8(num_str))
	bdi = DigitDraw(stroke_feeder=stroke_feeder, param=bdiprm)
	for digit in digits:
		digit_image = bdi.get(digit)
		digit_image.show()
	'''bdi = BasicDigitImage(param=bdiprm)
	di0 = bdi.get(0)
	di0.show()
	s_height = 30
	s_param = BasicDigitImage.calc_scale_from_height(s_height)
	digit_image_S = BasicDigitImage(s_param)
	multi_image_numbers = [0, 2]
	multi_number_image_size = (len(multi_image_numbers) * digit_image_S.size[0], digit_image_S.size[1])
	multi_number_image = Image.new('L', multi_number_image_size, (0,))
	x_offset = digit_image_S.size[0]
	offset = (0, 0)
	for n in multi_image_numbers:
		digit_image = digit_image_S.get(n)
		multi_number_image.paste(digit_image, offset)
		offset = offset[0] + x_offset, 0
	# digit_strokes_L = BasicDigitStrokes(scale=20, offset=(8, 8))
	l_height = 70
	l_param = BasicDigitImage.calc_scale_from_height(s_height)
	digit_image_L = BasicDigitImage(l_param)
	digit_image_L_0 = digit_image_L.get(0)
	digit_image_L_0.show()
	scale = 16
	digit_image_calc_result = DigitImage.calc_font_scale(scale=scale, line_width_ratio=0.25, padding_ratio=0.25)
	pp(digit_image_calc_result)
	digit_image = DigitImage()
	# scale, padding = digit_image.calc_scale_padding(scale=10, padding=2)
	for n in range(SEGPOINTS_MAX):
		# ofst_iw_ih = digit_image.calc_get_n(i)
		# pp(ofst_iw_ih)
		im = digit_image.get(n)
		im.show()
	sys.exit(0)
	get_0_str = 'im = digit_image.get_n(%d)'
	import cProfile
	cProfile.run(get_0_str % 0)
	print('2nd.0:')
	cProfile.run(get_0_str % 0)
	print('1:')
	cProfile.run(get_0_str % 1)
	print('2nd.0:')
	cProfile.run(get_0_str % 1)'''
