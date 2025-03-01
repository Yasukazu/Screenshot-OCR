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
from num_strokes import SEGPOINTS_MAX, DigitStrokes
from num_to_strokes import ImageFill

class DigitImage:
	def __init__(self, stroke_feeder=DigitStrokes(), cache_factor=1):
		self.stroke_feeder = stroke_feeder
		self.get_n: Callable[[int], Sequence[tuple[i_i_tpl, i_i_tpl]]] = lru_cache(maxsize=SEGPOINTS_MAX * cache_factor)(self._get_n)

	def _get_n(self, n: int, slant: StrokeSlant=StrokeSlant.SLANT02, scale=8, width_ratio=0.2, bgcolor=ImageFill.WHITE, padding=0.2)-> Image.Image:
		scale = round(scale * (1 + slant.value))
		ofst = round(scale * padding)
		img_w = round(scale + 2 * ofst)
		img_h = round(2 * scale + 2 * ofst)
		img = Image.new('L', (img_w, img_h), color=bgcolor.value)
		drw = ImageDraw.Draw(img)
		line_w = round(scale * width_ratio) or 1
		strokes = self.stroke_feeder.strokes(n, slant=slant, scale=scale, offset=(ofst, ofst))
		for stroke in strokes:
			drw.line(stroke, width=line_w, fill=ImageFill.invert(bgcolor).value)
		return img

if __name__ == '__main__':
	digit_image = DigitImage()
	img_0 = digit_image.get_n(0)
	img_0.show()
