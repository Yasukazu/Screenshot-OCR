from enum import IntEnum
from functools import wraps
from PIL import Image
from digit_image import BasicDigitImage, get_basic_number_image, BasicDigitImageParam
from format_num import HexFormatNum

class PutPos(IntEnum):
	L = -1
	C = 0
	R = 1

def put_number(pos: PutPos=PutPos.L, digit_image_feeder=BasicDigitImage(BasicDigitImage.calc_scale_from_height())):
	def _embed_number(func):
		@wraps(func)
		def wrapper(*ag, **kw):
			item_img: Image.Image = func(*ag, **kw)
			if item_img:
				number_str = kw['number_str']
				name_num_array = [HexFormatNum(int(n, 16)).conv_to_bin() for n in number_str]
				num_img = get_basic_number_image(name_num_array, digit_image_feeder=digit_image_feeder)
				x_offset = 0
				match pos:
					case PutPos.R:
						x_offset = item_img.width - num_img.width
				item_img.paste(num_img, (x_offset, 0))
				return item_img
		return wrapper
	return _embed_number

if __name__ == '__main__':
	from path_feeder import PathFeeder
	digit_image_param_S = BasicDigitImage.calc_scale_from_height(50)
	digit_image_feeder_S = BasicDigitImage(digit_image_param_S)
	path_feeder = PathFeeder()


	@put_number(pos=PutPos.L, digit_image_feeder=digit_image_feeder_S)
	def get_numbered_img(fn: str, number_str: str)-> Image.Image | None:
		fullpath = path_feeder.dir / (fn + path_feeder.ext)
		if fullpath.exists():
			img = Image.open(fullpath)
			return img
	n = '01'
	img = get_numbered_img(n, number_str=n)
	if img:
		img.show()
