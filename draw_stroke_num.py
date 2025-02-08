from collections import namedtuple
import numpy as np
from PIL import Image, ImageDraw

from num_to_strokes_pkl import get_stroke_list

stroke_list = get_stroke_list()

def get_stroke(n: int):
	assert 0 <= n < 16
	return stroke_list[n]

WH = namedtuple('WH', ['w', 'h'])

def draw_num(n: int, img: Image, offset: tuple[int, int]=(0, 0), percent: int=100, line_width_percent: int=25, fill=(0,)):
	'''offset:[width, height], fill:[R, G, B]'''
	drw = ImageDraw.Draw(img)
	ofst = WH(*offset)
	drw_wh = WH(*img.size)
	assert ofst.w < drw_wh.w and ofst.h + 1 < drw_wh.h
	w_scl = min(drw_wh.w - ofst.w, drw_wh.w * percent // 100 + line_width_percent - ofst.w)
	assert w_scl > 0
	h_scl = min(drw_wh.h - ofst.h, drw_wh.h * percent // 100 + line_width_percent - ofst.h) // 2
	assert h_scl > 0
	scl = min(w_scl, h_scl)
	line_width = scl * line_width_percent // 100
	offset = np.array(offset, int)
	strk = get_stroke(n)
	for stk in strk:
		seq = [tuple(st * scl + offset) for st in stk]
		drw.line(seq, fill=fill, width=line_width)

if __name__ == '__main__':
	from PIL import Image, ImageDraw
	percent = 40
	offset = np.array([20, 20], int)
	for i in range(10):
		img = Image.new('L', (80, 160), (0xff,))
		drw = ImageDraw.Draw(img)
		draw_num(i, img, offset=offset, percent=percent, line_width_percent=25)
		breakpoint()
		img.save(f"digi-{i}.png", 'PNG')
