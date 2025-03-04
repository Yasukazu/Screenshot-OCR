from collections import namedtuple
from math import floor
from functools import lru_cache
from collections.abc import Callable
from functools import cached_property
from collections.abc import Sequence
from enum import Enum
import pickle
import numpy as np
from PIL import ImageDraw, Image
from numpy.typing import NDArray
from pandas import DataFrame
from strok7 import STRK_DICT_STEM, StrokeSlant, i_i_tpl
from format_num import FormatNum, HexFormatNum, conv_num_to_bin
from num_strokes import SEGPOINTS_MAX, DigitStrokes, BasicDigitStrokes
from num_to_strokes import ImageFill

OfstIWIH = namedtuple('OfstIWIH', ['ofst', 'img_w', 'img_h'])
DigitImageCalcResult = namedtuple('DigitImageCalcResult', ['scale', 'font_scale', 'padding', 'line_width'])
class DigitImage:

	MIN_SCALE = 20
	STANDARD_PADDING = 1
	STANDARD_PADDING_RATIO = 0.25
	STANDARD_LINE_WIDTH = 2
	STANDARD_LINE_WIDTH_RATIO = 0.2

	def __init__(self, stroke_feeder=DigitStrokes(), slant: StrokeSlant=StrokeSlant.SLANT00, scale: int=MIN_SCALE, line_width=STANDARD_LINE_WIDTH, offset=(STANDARD_PADDING, STANDARD_PADDING), bgcolor=ImageFill.WHITE):
		self.stroke_feeder = stroke_feeder
		self.slant = slant
		self.scale = scale
		self.offset = offset
		self.line_width = line_width
		self.bgcolor = bgcolor
		self.get: Callable[[int], Sequence[tuple[i_i_tpl, i_i_tpl]]] = lru_cache(maxsize=SEGPOINTS_MAX)(self._get)

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
		strokes = self.stroke_feeder.strokes(n, slant=self.slant, scale=self.scale, offset=self.offset)
		for stroke in strokes:
			drw.line(stroke, width=self.line_width, fill=ImageFill.invert(self.bgcolor).value, joint='curve')
		return img


	def calc_scale_padding(self, scale=MIN_SCALE, padding=STANDARD_PADDING_RATIO, slant: StrokeSlant=StrokeSlant.SLANT02)-> tuple[int, int]:
		padding = scale * padding or 1
		_scale = scale - 2 * padding #  int(scale * (1 - slant.value))
		return int(_scale * (1 - slant.value)) + padding or 4 + padding, padding

class BasicDigitImage:
	def __init__(self, stroke_feeder:BasicDigitStrokes, line_width=2, bgcolor=ImageFill.WHITE):
		self.stroke_feeder = stroke_feeder
		self.line_width = line_width
		self.bgcolor = bgcolor
		self.get: Callable[[int], Sequence[tuple[i_i_tpl, i_i_tpl]]] = lru_cache(maxsize=SEGPOINTS_MAX)(self._get)

	def _get(self, n: int)-> Image.Image:
		img_w = self.stroke_feeder.scale + 2 * self.stroke_feeder.offset[0]
		img_h = 2 * self.stroke_feeder.scale + 2 * self.stroke_feeder.offset[1]
		img = Image.new('L', (img_w, img_h), color=self.bgcolor.value)
		drw = ImageDraw.Draw(img)
		strokes = self.stroke_feeder.get(n)
		for stroke in strokes:
			drw.line(stroke, width=self.line_width, fill=ImageFill.invert(self.bgcolor).value, joint='curve')
		return img


if __name__ == '__main__':
	import sys
	from pprint import pp
	from num_strokes import SEGPOINTS_MAX
	digit_strokes_S = BasicDigitStrokes(scale=10, offset=(4, 4))
	digit_image_S = BasicDigitImage(digit_strokes_S,line_width=2)
	digit_image_S_0 = digit_image_S.get(0)
	digit_image_S_0.show() 
	digit_strokes_L = BasicDigitStrokes(scale=20, offset=(8, 8))
	digit_image_L = BasicDigitImage(digit_strokes_L,line_width=4)
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
