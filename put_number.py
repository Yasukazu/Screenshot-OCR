from enum import IntEnum
from functools import wraps
from PIL import Image
from digit_image import BasicDigitImage #, get_basic_number_image #, BasicDigitImageParam
from segment_image import SegmentImage, get_hex_array_image
from format_num import HexFormatNum

class PutPos(IntEnum):
	L = -1
	C = 0
	R = 1
from typing import Any
def put_number(pos: PutPos=PutPos.L, digit_image_feeder=SegmentImage(BasicDigitImage.calc_scale_from_height())):
	'''prefix "0x" for hexadecimal'''
	def _embed_number(func):
		@wraps(func)
		def wrapper(*ag, **kw: dict[str, Any]):
			item_img: Image.Image = func(*ag, **kw)
			if item_img:
				number_str_list = str(kw['number_str']).split()
				name_num_array = bytearray()
				for number_str in number_str_list:
					d = int(number_str, 0)
					name_num_array += HexFormatNum(d).conv_to_bin()
				num_img = get_hex_array_image(name_num_array, image_feeder=digit_image_feeder)
				match pos:
					case PutPos.L:
						x_offset = 0
					case PutPos.R:
						x_offset = item_img.width - num_img.width
					case PutPos.C:
						x_offset = (item_img.width - num_img.width) // 2
					case _:
						raise ValueError("Illegal PutPos value!")
				item_img.paste(num_img, (x_offset, 0))
				return item_img
		return wrapper
	return _embed_number

if __name__ == '__main__':
	from path_feeder import PathFeeder
	digit_image_param_S = BasicDigitImage.calc_scale_from_height(50)
	digit_image_feeder_S = SegmentImage(digit_image_param_S)
	path_feeder = PathFeeder()

	@put_number(pos=PutPos.C, digit_image_feeder=digit_image_feeder_S)
	def get_numbered_img(fn: str, number_str: str)-> Image.Image | None:
		fullpath = path_feeder.dir / (fn + path_feeder.ext)
		if fullpath.exists():
			img = Image.open(fullpath)
			return img
	
	n = '01 -A'
	img = get_numbered_img(n.split()[0], number_str=n)
	if img:
		img.show()
