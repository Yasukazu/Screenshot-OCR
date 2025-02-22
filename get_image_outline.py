from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
from path_feeder import get_last_month_path
from pdf2image import convert_from_path
import numpy as np

def detect_edge(img_fullpath: Path):
	im = None
	if str(img_fullpath).endswith('.pdf') and img_fullpath.exists():
		imgs = convert_from_path(img_fullpath)
		im = imgs[0].convert('L')
	elif str(img_fullpath).endswith('.png') and img_fullpath.exists():
		im = Image.open(img_fullpath).convert('L')
	if not im:
		raise ValueError(f"{img_fullpath} is not supported.")
	# Detect edges and save
	edges = im.filter(ImageFilter.FIND_EDGES)
	h_size = edges.size[1]
	corner_img = edges.crop((0, 0, 2, h_size / 4))
	corner_array = np.array(corner_img)
	top_edge = 0
	for i, line in enumerate(corner_array[1:]):
		if line[1] > 0:
			top_edge = i
			break
	print(top_edge) # 249

def detect_tm_edge(img_fullpath: Path)-> int:
	assert str(img_fullpath).endswith('.png') and img_fullpath.exists()
	im = Image.open(img_fullpath).convert('L')
	corner_img = im.crop((0, 0, 1, im.height // 8))
	corner_array = np.array(corner_img)
	line_array = corner_array.transpose()[0]
	edge = list(line_array).index(255)
	draw = ImageDraw.Draw(im)
	draw.line((0, 0, 0, edge), fill=0, width=1)
	im.show()
	return edge


if __name__ == '__main__':
	# img_name = "2025-01-4p8.pdf"
	img_fullpath = get_last_month_path() / 'png' / '01.png'
	edge = detect_tm_edge(img_fullpath) # 77
	print(edge)
