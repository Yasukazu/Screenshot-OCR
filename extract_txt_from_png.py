from pathlib import Path
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
def get_top_text_img(input_path: Path)-> Image.Image:
	img = Image.open(input_path).convert('L')
	text_img = crop_top_text_image(img=img)
	return text_img

def get_np_img(input_path: Path)-> np.ndarray:
	img = Image.open(input_path).convert('L')
	return np.array(img)
CHECK_IMAGE = True
def crop_top_text_image(img: Image.Image)-> Image.Image:
	draw = ImageDraw.Draw(img)
	im = np.array(img)
	def get_white(ofst=0):
		for y in (range(ofst, img.height)):
			m = im[y]
			n = m[m == 255]
			if len(n) == img.width:
				return y
		return -1
	white_row = get_white()
	rr = []
	def draw_line(row)-> int:
		if CHECK_IMAGE:
			draw.line((0, row, img.width // 2, row), 0)
		return row
	rr += [draw_line(white_row)]
	def get_non_white(white_line):
		for y in (range(white_line, img.height)):
			m = im[y]
			n = m[m < 255]
			if len(n) == img.width:
				return y
		return -1
	non_white = get_non_white(white_row + 1)
	rr += [draw_line(non_white)]
	next_white = get_white(non_white + 1)
	rr += [draw_line(next_white)]
	next_non_white = get_non_white(next_white + 1) 
	rr += [draw_line(next_non_white)]
	
	def max_pitch():
		mx = 0
		max_pitch_r = 0
		last_r = 0
		for i, r in enumerate(rr):
			pitch = r - last_r
			if pitch > mx:
				mx = pitch 
				max_pitch_r = i
			last_r = r
		return max_pitch_r - 1
	max_pitch_pos = max_pitch()
	return img.crop((0, rr[max_pitch_pos], img.width, rr[max_pitch_pos + 1]))



		
def crop_bottom_text_image(img: Image.Image)-> Image.Image:
	im = np.array(img)
	def trm(y):
		return im[y:y + 1, 0:img.width]
	def first_dot():
		for y in range(img.height):
			m = trm(y)
			if not np.all(m):
				return y
	def first_white(ofst=0):
		for y in (range(img.height - ofst)):
			m = im[y]
			n = m[m == 255]
			if len(n):
				return y
		return -1
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

	first_white_line = first_white()
	draw.line((0, first_dot_line, img.width, first_dot_line), 0)
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

from path_feeder import PathFeeder, FileExt
from enum import StrEnum

class BtmOrTop(StrEnum):
    BTM = '.btm'
    TOP = '.top'

from typing import Iterator

def gen_btm_txt(sub_ext=BtmOrTop.BTM)-> Iterator[tuple[int, str]]:
	feeder = PathFeeder()
	#for input_path, output_path in path_feeder(3, 31):
	for stem in feeder.feed(padding=False):
		input_path = feeder.dir / (stem + feeder.ext)
		output_png_path = feeder.dir / (stem + sub_ext + feeder.ext)
		if (not output_png_path.exists()) or output_png_path.stat().st_size < 200:
			input_img = Image.open(input_path).convert('L')
			text_image = crop_bottom_text_image(input_img) if sub_ext == BtmOrTop.BTM else crop_top_text_image(input_img)
			text_image.save(output_png_path)
		output_btm_path = feeder.dir / (stem + sub_ext)
		output_btm_fullpath = feeder.dir / (stem + sub_ext + '.txt')
		if (not output_btm_fullpath.exists()) or output_btm_fullpath.stat().st_size == 0:
			run_cmd(output_png_path, output_btm_path)
		output_text = output_btm_fullpath.open().read()
		yield int(stem), output_text

from typing import Sequence

def write_btm_or_top_csv(input_type: FileExt=FileExt.BTM_TXT)-> Sequence[list[int, str]]:
	feeder = PathFeeder(input_type=input_type)
	output_path = feeder.dir / (input_type.value.ext[1:] + ".csv")
	rows = []
	import csv
	with output_path.open('w', encoding='utf8') as wcsv:
		writer = csv.writer(wcsv)
		for stem in feeder.feed(padding=False):
			input_path = feeder.dir / (stem + feeder.ext)
			text = input_path.open(encoding='utf8').read()
			ndnd = [s for s in text.split(' ' if input_type == FileExt.BTM_TXT else '\n') if s]
			nd = ''
			match(input_type):
				case FileExt.TOP_TXT:
					assert ndnd[-1] == 'この店舗の募集状況'
					nd = ''.join(ndnd[0:-1])
				case FileExt.BTM_TXT:
					assert ndnd[0] == '差引支給額'
					nd = ''.join(ndnd[1:])
			if not nd:
				raise ValueError('Failed to extract a valid text.')
			row = [int(stem)]
			match(input_type):
				case FileExt.TOP_TXT:
					row.append(nd)
				case FileExt.BTM_TXT:
					nums = ''.join([c for c in nd if c.isnumeric()])
					n = int(nums)
					row.append(n)
			writer.writerow(row)
			rows.append(row)
	return rows

if __name__ == '__main__':
	import sys
	from pprint import pp
	menu = 2
	match menu:
		case 1:
			for n, top_text in gen_btm_txt(sub_ext=BtmOrTop.TOP):
				print(n)
				print(top_text)
				print()
			sys.exit(0)
		case 2:
			rows = write_btm_or_top_csv(input_type=FileExt.TOP_TXT)
			pp(rows)
			sys.exit(0)
	sys.exit(0)
	feeder = PathFeeder() #input_type=FileExt.BTM_TXT)
	for stem in feeder.feed():
		break
	fullpath = feeder.dir / (stem + feeder.ext)
	top_text_image = get_top_text_img(fullpath)
	top_text_image.show()
	# gen_btm_txt()


