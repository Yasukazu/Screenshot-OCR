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
def get_misaki_digit_images():
	arc_font_fullpath = font_dir / arc_font_file_name
	image_list = {}
	if arc_font_fullpath.exists():
		page = Image.open(str(arc_font_fullpath))
		for n, page in enumerate(ImageSequence.Iterator(page)):
			image_list[n]=(page)
		return image_list
	font_file_name = 'misaki_gothic.png'
	font_fullpath =  font_dir / font_file_name
	if not font_fullpath.exists():
		raise ValueError(f"`{font_fullpath=}` does not exists!")
	full_image = Image.open(str(font_fullpath))
	ku_ten = np.array([16, 3], dtype=np.int64)
	end_point = ku_ten * 8
	offset = end_point - np.array([8, 10])
	area = np.array([offset, end_point])
	image_list = []
	for i in range(10):
		img_area = area.ravel().tolist()
		font_part = full_image.crop(img_area)
		font_part = font_part.resize([32, 40])
		area += np.array([8, 0])
		image_list.append(font_part)
	image_list[0].save(str(arc_font_fullpath),
		save_all=True, append_images=image_list[1:]) # compression='tiff_lzw'
	return {n: img for (n, img) in enumerate(image_list)}

if __name__ == '__main__':
	images = get_misaki_digit_images()
	for n, image in (images).items():
		print(n)
		image.show()