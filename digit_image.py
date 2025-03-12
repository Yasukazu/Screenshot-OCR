from enum import Enum
from collections import namedtuple
from functools import lru_cache
from collections.abc import Callable
from collections.abc import Sequence
from PIL import ImageDraw, Image
from strok7 import STRK_DICT_STEM, StrokeSlant, i_i_tpl
from format_num import FormatNum, HexFormatNum, conv_num_to_bin, formatnums_to_bytearray
from num_strokes import SEGPOINTS_MAX, DigitStrokes, BasicDigitStrokes

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


BasicDigitImageParam = namedtuple('BasicDigitImageParam', ['scale', 'padding', 'line_width'])

BASIC_DIGIT_IMAGE_PARAM_LIST = [
	BasicDigitImageParam(scale=8, padding=(2, 2), line_width=2),
	BasicDigitImageParam(scale=16, padding=(4, 4), line_width=4),
]
class BasicDigitImage:
	WIDTH = 8
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
	def calc_scale_from_width(cls, width: int=WIDTH, padding: int=-1, line_width: int=-1)-> BasicDigitImageParam:
		if padding < 0:
			padding = width // 4
		if line_width < 0:
			line_width = width // 8 or 1
		offset = padding + line_width // 2 or 1
		scale = width - 2 * offset
		return BasicDigitImageParam(scale=scale, padding=(padding, padding), line_width=line_width)

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

	def __init__(self, width: int, scale: int, padding: Sequence[int], line_width: int,
			slant: StrokeSlant=StrokeSlant.SLANT00, 
			bgcolor=ImageFill.WHITE):
		stroke_scale = scale # - padding[0] * 2 - line_width
		stroke_offset = [pad + line_width // 2 or 1 for pad in padding]
		self.stroke_feeder = digit_strokes.DigitStrokes(slant=slant, scale=stroke_scale, offset=(stroke_offset[0], stroke_offset[1]))
		self.line_width = line_width
		self.bgcolor = bgcolor
		self.size = width, width * 2
		self.get: Callable[[int], Image.Image] = lru_cache(maxsize=self.stroke_feeder.get_max())(self._get)

	def _get(self, n: int)-> Image.Image:
		# img_w = self.scale # stroke_feeder.scale + 2 * self.stroke_feeder.offset[0] + self.line_width
		# img_h = 2 * self.scale # stroke_feeder.scale + 2 * self.stroke_feeder.offset[1] + self.line_width
		img = Image.new('L', self.size, color=self.bgcolor.value)
		drw = ImageDraw.Draw(img)
		strokes = self.stroke_feeder.get(n)
		for stroke in strokes:
			drw.line(stroke, width=self.line_width, fill=ImageFill.invert(self.bgcolor).value, joint='curve')
		return img

def get_basic_number_image(nn: Sequence[int | FormatNum] | bytearray, digit_image_feeder=BasicDigitImage(BasicDigitImage.WIDTH, *BasicDigitImage.calc_scale_from_width()))-> Image.Image:
	b_array = nn if isinstance(nn, bytearray) else formatnums_to_bytearray(nn)
	number_image_size = len(b_array) * digit_image_feeder.size[0], digit_image_feeder.size[1]
	number_image = Image.new('L', number_image_size, (0,))
	offset = (0, 0)
	x_offset = digit_image_feeder.size[0]
	for n in b_array:
		digit_image = digit_image_feeder.get(n)
		number_image.paste(digit_image, offset)
		offset = offset[0] + x_offset, 0
	return number_image

if __name__ == '__main__':
	import sys
	from pprint import pp
	width = 24
	bdiprm = BasicDigitImage.calc_scale_from_width(width)
	bdi = BasicDigitImage(width, *bdiprm)
	di0 = bdi.get(0)
	di0.show()
	s_width = 16
	s_param = BasicDigitImage.calc_scale_from_width(s_width)
	digit_image_S = BasicDigitImage(s_width, *s_param)
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
	l_width = 32
	l_param = BasicDigitImage.calc_scale_from_width(s_width)
	digit_image_L = BasicDigitImage(l_width, *l_param)
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
	cProfile.run(get_0_str % 1)
