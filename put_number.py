from enum import IntEnum
from functools import wraps
from PIL import Image
from digit_image import BasicDigitImage, get_basic_number_image
from format_num import HexFormatNum

class PutPos(IntEnum):
	L = -1
	C = 0
	R = 1

def put_number(pos: PutPos=PutPos.L, digit_image_feeder=BasicDigitImage()):
	def _embed_number(func):
		@wraps(func)
		def wrapper(*ag, **kw):
			item_img: Image.Image = func(*ag, **kw)
			if item_img and kw['number_str']:
				name_num_array = HexFormatNum.conv_to_bin(kw['number_str'])
				num_img = get_basic_number_image(name_num_array, digit_image_feeder=digit_image_feeder)
				x_offset = 0
				match pos:
					case PutPos.R:
						x_offset = item_img.width - num_img.width
				item_img.paste(num_img, (x_offset, 0))
				return item_img
		return wrapper
	return _embed_number
