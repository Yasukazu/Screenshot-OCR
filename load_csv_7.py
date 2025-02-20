import csv
import pickle
import numpy as np
import numpy.typing as npt

def conv(i: str)-> bool:
	return False if i == '' else bool(int(i))
class Seg7:
	def __init__(self, a: str, b: str, c: str, d: str, e: str, f: str, g: str, h: str):
		self.a: bool = conv(a)
		self.b: bool = conv(b)
		self.c: bool = conv(c)
		self.d: bool = conv(d)
		self.e: bool = conv(e)
		self.f: bool = conv(f)
		self.g: bool = conv(g)
		self.h: int = int(h)

PICKLE_EXT = '.pkl'
from strok7 import SegLine

from seven_seg import SEVEN_SEG_STEM
# return np.loadtxt(fname, delimiter=',')
# with open('7-seg.csv', encoding='utf8') as csv_file:
#	csv_reader = csv.reader(csv_file)
	#next(csv_reader)  # skipping the header
def _load_seg7(fname: str=SEVEN_SEG_STEM + '.csv'):
	with open(fname, encoding='ASCII') as f:
		reader = csv.reader(f)
		csv_7_seg_list = [r for r in reader]
	seg7_list = []
	for i, row in enumerate(csv_7_seg_list):
		seg = Seg7(*row)
		assert seg.h == i
		seg7_list.append(seg)
	assert len(seg7_list) == 16
	return seg7_list

def get_seg7_list():
	from num_to_strokes import load_seg7
	return load_seg7()

seg7_list = get_seg7_list()
from seven_seg import SEVEN_SEG_SIZE
assert len(seg7_list) == SEVEN_SEG_SIZE

import numpy as np

NP_STRK_DICT_PKL = 'np_strk_dict.pkl'
def load_np_strk_dic(pklf=NP_STRK_DICT_PKL):
	with open(pklf, 'rb') as buf:
		return pickle.load(buf)

from strok7 import get_np_strk_dict
np_strk_dict = get_np_strk_dict() # load_np_strk_dic() # {k: np.array(v, int) for (k, v) in strk_dic.items() }

def get_strok(n):
	seg7 = seg7_list[n]
	seg7_dict = seg7.__dict__
	strks = []
	for k in 'abcdefg':
		if seg7_dict[k]:
			strks.append(np_strk_dict[k])
	return strks


_seg7_set_list = []
def load_seg7_set_list():
	from num_to_strokes import get_seg7_list
	seg7_set_list = get_seg7_list()
	for seg_set in seg7_set_list:
		_seg7_set_list.append(np.array([seg.value for seg in seg_set]))
load_seg7_set_list()
def get_seg_lines(n: int):
	return _seg7_set_list[n]
_segelem7dict = {}


if __name__ == '__main__':
	from num_to_strokes import draw_digit
	from PIL import Image, ImageDraw
	from pprint import pprint
	save = False
	scale = 40
	offset = np.array([20, 20], int)
	for i in range(10):
		img = Image.new('L', (80, 160), (0xff,))
		drw = ImageDraw.Draw(img)
		draw_digit(i, drw, offset=offset, scale=scale, line_width_ratio=8)
		img.show()
		if save:
			img.save(f"digi-{i}.png", 'PNG')
