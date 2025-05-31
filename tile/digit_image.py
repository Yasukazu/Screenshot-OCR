from enum import Enum
from collections import namedtuple
from functools import lru_cache
from dataclasses import dataclass, field
from collections.abc import Callable
from collections.abc import Sequence
from PIL import ImageDraw, Image
from image_fill import ImageFill
from format_num import FormatNum, FormatNum, formatnums_to_bytearray
from num_strokes import SEGPOINTS_MAX #, DigitStrokes, BasicDigitStrokes

from seg_node_pair import DigitStrokeFeeder, encode_str_to_seg7bit8
from seg7bit8 import Seg7Bit8
from digit_strokes import DigitStrokes


OfstIWIH = namedtuple('OfstIWIH', ['ofst', 'img_w', 'img_h'])
DigitImageCalcResult = namedtuple('DigitImageCalcResult', ['scale', 'font_scale', 'padding', 'line_width'])

@dataclass
class DigitImageParam:
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

class DigitImage:

	MIN_SCALE = 20
	STANDARD_PADDING = 1
	STANDARD_PADDING_RATIO = 0.25
	STANDARD_LINE_WIDTH = 2
	STANDARD_LINE_WIDTH_RATIO = 0.2

	@classmethod
	def calc_scale_from_height(cls, height: int, padding: int=-1, line_width: int=-1)-> DigitImageParam:
		if padding < 0:
			padding = height // 8
		if line_width < 0:
			line_width = height // 10 or 1
		scale = (height - 2 * padding - line_width) // 2 
		if scale - line_width <= 0:
			raise ValueError(f"scale({scale}) is too small to show!")
		width = scale + 2 * padding + line_width
		return DigitImageParam(width=width, scale=scale, padding=(padding, padding), line_width=line_width)

	def __init__(self, stroke_feeder: DigitStrokes, bgcolor=ImageFill.WHITE, line_width=STANDARD_LINE_WIDTH): # slant: StrokeSlant=StrokeSlant.SLANT00, scale: int=MIN_SCALE,  offset=(STANDARD_PADDING, STANDARD_PADDING)#=BasicDigitStrokes()
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

	@property
	def size(self):
		img_w = self.stroke_feeder.scale + 2 * self.stroke_feeder.offset[0]
		img_h = 2 * self.stroke_feeder.scale + 2 * self.stroke_feeder.offset[1]
		return img_w, img_h

	def _get(self, sb: int)-> Image.Image: # Seg7Bit8
		img = Image.new('L', self.size, color=self.bgcolor.value)
		drw = ImageDraw.Draw(img)
		strokes = self.stroke_feeder.get(sb)
		for st in strokes:
			if st[0] and st[1]:
				ft = st[0], st[1]
				drw.line(ft, width=self.line_width, fill=ImageFill.invert(self.bgcolor).value, joint='curve')
		return img


	'''def calc_scale_padding(self, scale=MIN_SCALE, padding=STANDARD_PADDING_RATIO, slant: StrokeSlant=StrokeSlant.SLANT02)-> tuple[int, float]:
		padding = scale * padding or 1
		_scale = scale - 2 * padding #  int(scale * (1 - slant.value))
		return int(_scale * (1 - slant.value) * padding or 4 * padding), padding
	'''


class BasicDigitImage:
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

	def __init__(self, stroke_feeder: DigitStrokeFeeder, param: DigitImageParam,
			bgcolor=ImageFill.WHITE): # width: int, scale: int, padding: Sequence[int], line_width: int slant: StrokeSlant=StrokeSlant.SLANT00, 
		#if (not stroke_feeder) and (not param):raise ValueError("Needs stroke_feeder or param!")
		# stroke_scale = param.scale # - padding[0] * 2 - line_width
		# stroke_offset = [pad + param.line_width // 2 or 1 for pad in param.padding]
		self.stroke_feeder = stroke_feeder # digit_strokes.DigitStrokes(slant=slant, scale=stroke_scale, offset=(stroke_offset[0], stroke_offset[1]))
		self.param = param
		# self.line_width = param.line_width
		self.bgcolor = bgcolor
		# self.size = param.width, param.height
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
		digit_strokes = self.stroke_feeder.get_digit(s)
		for strokes in digit_strokes:
			for stroke in strokes:
				drw.line(stroke, width=self.line_width, fill=ImageFill.invert(self.bgcolor).value, joint='curve')
		return img

def get_basic_number_image(nn: Sequence[int | FormatNum] | bytearray, digit_image_feeder: BasicDigitImage)-> Image.Image: #BasicDigitImage.WIDTH, (param=BasicDigitImage.calc_scale_from_height(BasicDigitImage.HEIGHT))
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
	bdi_param = BasicDigitImage.calc_scale_from_height(height)
	digit_strokes = DigitStrokes(scale=bdi_param.scale, offset=bdi_param.padding)
	stroke_feeder = DigitStrokeFeeder(digit_strokes=digit_strokes, scale=bdi_param.scale, offset=bdi_param.padding)
	num = 1.23
	num_str = "%.2f" % num
	digits = list(encode_str_to_seg7bit8(num_str))
	bdi = DigitImage(stroke_feeder=digit_strokes, line_width=bdi_param.line_width)
	zero_to_9 = [c for c in '0123456789']
	c_to_int = {c:n for (n, c) in enumerate(zero_to_9)}
	c_to_int['.'] = 16
	for c in num_str:
		digit = c_to_int[c]
		digit_image = bdi.get(digit)
		digit_image.show()

