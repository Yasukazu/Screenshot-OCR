# Misaki font
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image, ImageSequence
from dotenv import load_dotenv
load_dotenv()
import sys,os
from pathlib import Path

screen_base_dir_name = os.getenv('SCREEN_BASE_DIR')

if not screen_base_dir_name:
	raise ValueError(f"{screen_base_dir_name=} is not set in env.!")

screen_dir = Path(screen_base_dir_name)
font_dir = Path(screen_base_dir_name) / 'font'

if not font_dir.exists():
	raise ValueError(f"`{font_dir=}` does not exists!")


FONT_SIZE = (8, 8)
HALF_FONT_SIZE = (4, 8)

# type ku = tuple[tuple[int, int], int]

DIGIT_KU = (3, 16),10
HEX_KU = (3, 33),6
TEN_KU = (1, 4),2

def array(seq):
	return np.array(seq, np.int64)

font_size_array = array(FONT_SIZE)
font_file_name = 'misaki_gothic.png'

PNG_EXT = '.png'

class Byte(int):
	@property
	def h(self):
		return self >> 4
	@property
	def l(self):
		return self & 0xf

def byte_to_xy(byte: int):
	b = Byte(byte)
	return b.l, b.h

class FontNameSize(Enum):
	misaki_4x8 = (64, 128)
	misaki_mincho = (752, 752)

class MisakiFont(Enum):
	half_font = FontNameSize.misaki_4x8
	HALF_NAME = FontNameSize.misaki_4x8.name
	HALF_SIZE = FontNameSize.misaki_4x8.value
	full_font = FontNameSize.misaki_mincho
	FULL_NAME = FontNameSize.misaki_4x8.name
	FULL_SIZE = FontNameSize.misaki_4x8.value

	@classmethod
	def get_font_dir(cls):
		return font_dir

	@classmethod
	def get_font_fullpath(cls, font=FontNameSize.misaki_4x8):
		font_file_name = font.name + PNG_EXT
		font_fullpath = font_dir / font_file_name
		return font_fullpath

	@classmethod
	def get_font_base_image(cls, font=FontNameSize.misaki_4x8)-> Image.Image:
		font_file_name = font.name + PNG_EXT
		font_fullpath = font_dir / font_file_name
		if not font_fullpath.exists():
			raise ValueError(f"`{font_fullpath=}` does not exists!")
		base_image = Image.open(str(font_fullpath))
		return base_image

	@classmethod
	def get_half_font_image(cls, c: int, font_dict: dict[int, Image.Image]={}):
		if c > 255 or c < 0:
			raise ValueError('Must be an unsigned byte(0 to 255)!')
		if c in font_dict:
			return font_dict[c]
		font_file_name = MisakiFont.HALF_NAME.value + PNG_EXT
		font_fullpath = font_dir / font_file_name
		if not font_fullpath.exists():
			raise ValueError(f"`{font_fullpath=}` does not exists!")
		full_image = Image.open(str(font_fullpath))
			
		def append_fonts(ku: int): 
			x_pos, y_pos = byte_to_xy(ku)
			x_offset = x_pos * 4
			y_offset = y_pos * 8
			offset = array([x_offset, y_offset])
			end_point = offset + np.array(HALF_FONT_SIZE)
			area = array([offset, end_point])
			img_area = area.ravel().tolist()
			font_part = full_image.crop(img_area)
			return font_part
		font_image = append_fonts(c)
		font_dict[c] = font_image
		return font_image

	@classmethod
	def get_line_images(cls, font=FontNameSize.misaki_4x8):
		base_image = cls.get_font_base_image(font)
		images = []	
		height = font.value[1]
		line_image_size = (font.value[0], 8)
		box = [0, 0, *line_image_size]
		def shift():
			box[1] += 8
			box[3] += 8
		for n, line in enumerate(range(height // 8)):
			images.append(base_image.crop(box))
			shift()
		return images

class SecondMisakiFont(Enum):
	HALF_NAME = 'misaki_gothic_2nd_4x8'
	FULL_NAME = 'misaki_gothic_2nd'

arc_font_file_name = 'misaki_gothic-digit.tif'

def get_misaki_digit_images(scale=1):
	re_size = (array(FONT_SIZE) * scale).tolist() if scale > 1 else None
	arc_font_fullpath = font_dir / arc_font_file_name
	image_list = {}
	if arc_font_fullpath.exists():
		page = Image.open(str(arc_font_fullpath))
		for n, page in enumerate(ImageSequence.Iterator(page)):
			if re_size:
				page = page.resize(re_size)
			image_list[n]=(page)
		return image_list
	font_fullpath = font_dir / font_file_name
	if not font_fullpath.exists():
		raise ValueError(f"`{font_fullpath=}` does not exists!")
	full_image = Image.open(str(font_fullpath))
	def kuten_to_xy(*kuten: int):
		return kuten[1] - 1, kuten[0] - 1
	image_list = []
	def append_fonts(ku): 
		num_ku_ten, num_range = ku
		num_pos = kuten_to_xy(*num_ku_ten)
		num_addr = array(num_pos)
		offset = (num_addr * 8) # - array([0, 2])
		end_point = offset + np.array(FONT_SIZE)
		area = array([offset, end_point])
		for i in range(num_range):
			img_area = area.ravel().tolist()
			font_part = full_image.crop(img_area)
			#font_part = font_part.resize([32, 40])
			area += np.array([8, 0])
			image_list.append(font_part)
	append_fonts(DIGIT_KU)
	append_fonts(HEX_KU)
	append_fonts(TEN_KU)
	image_list[0].save(str(arc_font_fullpath),
		save_all=True, append_images=image_list[1:]) # compression='tiff_lzw'
	return {n: img for (n, img) in enumerate(image_list)}

class MisakiFontImage:
	def __init__(self, scale=1):
		self.digit_fonts = get_misaki_digit_images(scale=scale)
		self.scale = scale
	def get_number_image(self, num_array: bytearray):
		scaled = font_size_array * self.scale
		image_list = [pil2cv(self.digit_fonts[b]) for b in num_array]
		n_img = cv2.hconcat(image_list)
		n_img[n_img == 1] = 255
		pil_number_image = cv2pil(n_img)
		return pil_number_image
	'''
		image_fullpath = screen_dir / 'number_image.png'
		cv2.imwrite(str(image_fullpath), n_img)
		for n, b in enumerate(num_array):
			font_img = pil2cv(self.fonts[b])
			cv2.namedWindow("image", cv2.WINDOW_NORMAL)
			cv2.imshow("image", font_img)
			image.paste(self.fonts[b], (w * n, 0))'''


def pil2cv(image: Image):
	''' PIL型 -> OpenCV型 '''
	new_image = np.array(image, dtype=np.uint8)
	if new_image.ndim == 2:  # モノクロ
		pass
	elif new_image.shape[2] == 3:  # カラー
		new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
	elif new_image.shape[2] == 4:  # 透過
		new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
	return new_image


def cv2pil(image):
	''' OpenCV型 -> PIL型 '''
	new_image = image.copy()
	if new_image.ndim == 2:  # モノクロ
		new_image[new_image == 1] = 255
		#return new_image.convert('L')
	elif new_image.shape[2] == 3:  # カラー
		new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
	elif new_image.shape[2] == 4:  # 透過
		new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
	new_image = Image.fromarray(new_image)
	return new_image

if __name__ == '__main__':
	import sys, os
	font = MisakiFont.half_font.value
	line_images = MisakiFont.get_line_images(font)
	arc_fullpath = MisakiFont.get_font_dir() / (font.name + '.tif')
	sys.exit(0)

	b = sys.argv[1] # b = input('char for the font:')
	c = ord(b)
	if c > 255:
		raise ValueError(f"{c=} exceeds half font range(255)")
	font_image = MisakiFont.get_half_font_image(c)
	if font_image:
		font_image.show()
	sys.exit(0)
	from format_num import formatnums_to_bytearray, FloatFormatNum
	mfi = MisakiFontImage(8)
	fn = [FloatFormatNum(3.14)]
	ba = formatnums_to_bytearray(fn, conv_to_bin2=False)
	ni = mfi.get_number_image(ba)
	ni.show()
	images = get_misaki_digit_images(8)
	a_img = images[0xa]
	f_img = images[0xf]
	m_img = images[0xf + 1]
	p_img = images[0xf + 2]
	for n, image in (images).items():
		print(n)
		#image.show()