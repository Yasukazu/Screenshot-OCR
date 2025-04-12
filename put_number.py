from enum import IntEnum
from functools import wraps
from PIL import Image
#from digit_image import BasicDigitImage #, get_basic_number_image #, BasicDigitImageParam
#from segment_image import SegmentImage, get_number_image
from format_num import HexFormatNum

class PutPos(IntEnum):
	L = -1
	C = 0
	R = 1
from typing import Any
from misaki_font import MisakiFontimage
NUMBER_STR = 'number_str'
NUMBER = 'number'
def put_number(pos: PutPos=PutPos.L, digit_image_feeder=MisakiFontimage):#SegmentImage(BasicDigitImage.calc_scale_from_height())):
	'''prefix "0x" for hexadecimal'''
	from format_num import formatnums_to_bytearray
	def _embed_number(func):
		@wraps(func)
		def wrapper(*ag, **kw: dict[str, Any]):
			item_img: Image.Image = func(*ag, **kw)
			if item_img:
				if NUMBER_STR in kw:
					number_str_list = str(kw[NUMBER_STR]).split()
					bb = bytearray()
					for number_str in number_str_list:
						d = int(number_str, 0)
						bb += HexFormatNum(d).conv_to_bin()
				elif NUMBER in kw:
					bb = formatnums_to_bytearray(kw[NUMBER], conv_to_bin2=False)
				num_img = digit_image_feeder.get_number_image(bb)
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
	from path_feeder import DbPathFeeder
	from misaki_font import MisakiFontimage
	#digit_image_param_S = BasicDigitImage.calc_scale_from_height(50)
	digit_image_feeder_S = MisakiFontimage(8) #SegmentImage(digit_image_param_S)
	path_feeder = DbPathFeeder()
	from pathlib import Path
	@put_number(pos=PutPos.C, digit_image_feeder=digit_image_feeder_S)
	def get_numbered_img(fullpath: Path, number: float)-> Image.Image | None:
		if fullpath.exists():
			img = Image.open(str(fullpath))
			return img
	import sys
	n = float(sys.argv[1])#'01 -A'
	for day_stem in path_feeder.feed():
		break
	month = path_feeder.month
	date_num_str = f"{month}.{day_stem[0]:02}"
	date_num = float(date_num_str)
	fullpath = path_feeder.dir / day_stem[1]
	img = get_numbered_img(fullpath, number=date_num) # n.split()[0]
	if img:
		img.show()
