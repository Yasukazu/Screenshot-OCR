# Misaki font
import numpy as np
from PIL import Image, ImageSequence
from dotenv import load_dotenv
load_dotenv()
import sys,os
from pathlib import Path
screen_base_dir_name = os.getenv('SCREEN_BASE_DIR')
if not screen_base_dir_name:
	raise ValueError(f"{screen_base_dir_name=} is not set in env.!")
font_dir = Path(screen_base_dir_name) / 'font'
if not font_dir.exists():
	raise ValueError(f"`{font_dir=}` does not exists!")
arc_font_file_name = 'misaki_gothic-digit.tif'
FONT_SIZE = (8, 8)
type ku = tuple[tuple[int, int], int]
DIGIT_KU: ku = (3, 16),10
HEX_KU: ku = (3, 33),6
TEN_KU: ku = (1, 4),2
def array(seq):
	return np.array(seq, np.int64)
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
	font_file_name = 'misaki_gothic.png'
	font_fullpath =  font_dir / font_file_name
	if not font_fullpath.exists():
		raise ValueError(f"`{font_fullpath=}` does not exists!")
	full_image = Image.open(str(font_fullpath))
	def kuten_to_xy(*kuten: int):
		return kuten[1] - 1, kuten[0] - 1
	image_list = []
	def append_fonts(ku: ku): 
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

if __name__ == '__main__':
	images = get_misaki_digit_images(8)
	a_img = images[0xa]
	f_img = images[0xf]
	m_img = images[0xf + 1]
	p_img = images[0xf + 2]
	for n, image in (images).items():
		print(n)
		#image.show()