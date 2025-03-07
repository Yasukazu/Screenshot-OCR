import subprocess

cmd = 'tesseract' # -l jpn+eng 

def run_cmd(input_path, output_txt_path, lang='jpn+eng'):
	try:
		# OCRmyPDF command with optimization options
		command = [cmd, '-l', lang, input_path, output_txt_path] # '--pdf-renderer', 'hocr', '--optimize', '0', 
		
		# Execute the OCRmyPDF command
		subprocess.run(command, check=True)
		
		print(f" file:{output_txt_path} is generated from:{input_path}")
	except subprocess.CalledProcessError as e:
		print(f"tesseract error: {e}")
		
from pathlib import Path
import numpy as np
def trim(array, x, y, width, height):
    return array[y:y + height, x:x+width]
from PIL import Image, ImageDraw, ImageOps
def crop_bottom_text_image(input_path: Path)-> Image.Image:
	img = Image.open(input_path).convert('L')
	im = np.array(img)
	def trm(y):
		return im[y:y + 1, 0:img.width]
	def first_dot():
		for y in range(img.height):
			m = trm(y)
			if not np.all(m):
				return y
	def bottom_dot(ofst=0):
		for y in reversed(range(img.height - ofst)):
			m = im[y]
			n = m[m < 255]
			if len(n):
				return y
		return -1
	def bottom_dot_next(ofst: int):
		for y in reversed(range(ofst)):
			m = im[y]
			if (m == 255).all():
				break
		return y

	#first_dot_line = first_dot()
	#draw.line((0, first_dot_line, img.width, first_dot_line), 0)
	under_text = bottom_dot()
	over_text = bottom_dot_next(under_text - 1)
	padding = 5
	return img.crop((0, over_text - padding, img.width, under_text + padding))
	# draw.line((0, dot_line, img.width, dot_line), 0)
	# bottom_blank_line = bottom_blank2(last_dot_line)
	gmi = ImageOps.flip(img)
	mi = np.array(gmi) # flipud(im)
	first_dot_line_of_flip = first_dot()
	def any_non_zero():
		return np.any(mi!=0, axis=0)
	row_idx = any_non_zero()
	bottom_right = img.size
	bottom_left = 0, img.size[1]
	offset = 50
	line_from = 0, bottom_left[1] - offset
	line_to = img.size[0], bottom_left[1] - offset
	draw.line((line_from, line_to), (0,))
	img.show()


import os
from pathlib import Path

def path_feeder(from_=1, to=31, input_ext='.png', output_ext='.tact'): #rng=range(0, 31)):
	home_dir = os.path.expanduser('~')
	home_path = Path(home_dir)
	input_dir = home_path / 'Documents' / 'screen' / '202501'
	assert input_dir.exists()
	for day in range(from_, to + 1):
		input_filename = f'2025-01-{day:02}{input_ext}'
		input_fullpath = input_dir / input_filename
		if not input_fullpath.exists():
			continue
		input_path_noext, _ext = os.path.splitext(input_fullpath)
		output_path = Path(input_path_noext + output_ext)
		assert not output_path.exists()
		yield input_fullpath, output_path

if __name__ == '__main__':
	import os, sys
	from path_feeder import PathFeeder
	feeder = PathFeeder()
	#for input_path, output_path in path_feeder(3, 31):
	for stem in feeder.feed(padding=False):
		input_path = feeder.dir / (stem + feeder.ext)
		output_path = feeder.dir / stem # '.txt')
		output_png_path = feeder.dir / (stem + '.btm' + feeder.ext)
		if not output_png_path.exists():
			text_image = crop_bottom_text_image(input_path)
			text_image.save(output_png_path)
		output_btm_path = feeder.dir / (stem + '.btm')
		if not (feeder.dir / (stem + '.btm' + '.txt')).exists():
			run_cmd(output_png_path, output_btm_path)