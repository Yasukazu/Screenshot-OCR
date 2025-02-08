import csv
import pickle

def conv(i):
	return 0 if i == '' else int(i)
class Seg7:
	def __init__(self, a, b, c, d, e, f, g, h):
		self.a = conv(a)
		self.b = conv(b)
		self.c = conv(c)
		self.d = conv(d)
		self.e = conv(e)
		self.f = conv(f)
		self.g = conv(g)
		self.h = conv(h)

# with open('7-seg.csv', encoding='utf8') as csv_file:
#	csv_reader = csv.reader(csv_file)
	#next(csv_reader)  # skipping the header
def load_seg7(pkl='7-seg.pkl'):
	seg7_list = []
	rf = open(pkl, 'rb').read()
	csv_reader = pickle.loads(rf)
	for i, row in enumerate(csv_reader):
		seg = Seg7(*row)
		assert seg.h == i
		seg7_list.append(seg)
	assert len(seg7_list) == 16
	return seg7_list

seg7_list = load_seg7()

assert len(seg7_list) == 16

class Digi7:
	dic = [(0,)] * 16
	def __init__(self, s: Seg7):
		self.dic[seg.h] = (s.a, s.b, s.c, s.d, s.e, s.f, s.g)

for seg in seg7_list:
	Digi7(seg)
digi7_list = [
	Digi7(seg) for seg in seg7_list
]


import numpy as np

NP_STRK_DICT_PKL = 'np_strk_dict.pkl'
def load_np_strk_dic(pklf=NP_STRK_DICT_PKL):
	with open(pklf, 'rb') as buf:
		return pickle.load(buf)

np_strk_dict = load_np_strk_dic() # {k: np.array(v, int) for (k, v) in strk_dic.items() }

def get_strok(n):
	seg7 = seg7_list[n]
	seg7_dict = seg7.__dict__
	strks = []
	for k in 'abcdefg':
		if seg7_dict[k]:
			strks.append(np_strk_dict[k])
	return strks

from num_to_strokes_pkl import get_stroke_list
stroke_list = get_stroke_list()

def get_stroke(n: int):
	assert 0 <= n < 16
	return stroke_list[n]

from PIL import Image, ImageDraw

def draw_num(n, drw, offset=(0,0), scale=16, width=8, fill=(0,)):
	offset = np.array(offset, int)
	strk = get_stroke(n)
	for stk in strk:
		seq = [tuple(st * scale + offset) for st in stk]
		drw.line(seq, fill=fill, width=width)

if __name__ == '__main__':
	from pprint import pprint
	scale = 40
	offset = np.array([20, 20], int)
	for i in range(10):
		img = Image.new('L', (80, 160), (0xff,))
		drw = ImageDraw.Draw(img)
		draw_num(i, drw, offset=offset, scale=scale, width=8)

		img.save(f"digi-{i}.png", 'PNG')

	pprint(np_strk_dict)
	pprint(Digi7.dic)
	for u in seg7_list:
		print(u.a, u.b, u.c, u.d, u.e, u.f, u.g, u.h)
from PIL import Image, ImageDraw

def draw_num(n, drw, offset=(0,0), scale=16, width=8, fill=(0,)):
	offset = np.array(offset, int)
	strk = get_strok(n)
	for stk in strk:
		seq = [tuple(st * scale + offset) for st in stk]
		drw.line(seq, fill=fill, width=width)

if __name__ == '__main__':
	from pprint import pprint
	scale = 40
	offset = np.array([20, 20], int)
	for i in range(10):
		img = Image.new('L', (80, 160), (0xff,))
		drw = ImageDraw.Draw(img)
		draw_num(i, drw, width=20)

		img.save(f"digi-{i}.png", 'PNG')

	pprint(np_strk_dict)
	pprint(Digi7.dic)
	for u in seg7_list:
		print(u.a, u.b, u.c, u.d, u.e, u.f, u.g, u.h)