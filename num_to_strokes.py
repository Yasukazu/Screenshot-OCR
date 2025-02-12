import pickle
from strok7 import STRK_DICT_STEM

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
		self.h: bool = conv(h)

SEVEN_SEG_STEM = '7-seg'
PICKLE_EXT = '.pkl'
SEVEN_SEG_MAX = 15

from strok7 import SegLine

_seg7_list: list[set[Seg7]] = [set()] * 16

def load_seg7()-> list[set[SegLine]]:
	from load_csv_7 import load_7_seg_num_csv_as_df
	df = load_7_seg_num_csv_as_df()
	for i in range(16):
		dline = df[i: i + 1]
		seg_set = set()
		for c in 'abcdefg':
			if int(list(dline[c])[0]) > 0:
				seg = SegLine.get(c)
				seg_set.add(seg) # dlst.append(seg) # '1') # SegFlag.get(c).value)
		_seg7_list[i] = seg_set
	return _seg7_list

load_seg7()

def get_seg7_list():
	return _seg7_list

def _load_seg7(pkl: str=SEVEN_SEG_STEM + PICKLE_EXT):
	rf = open(pkl, 'rb').read()
	csv_reader = pickle.loads(rf)
	for i, row in enumerate(csv_reader):
		seg = Seg7(*row)
		assert seg.h == i
		_seg7_list.append(seg)
	assert len(_seg7_list) == 16
	return _seg7_list

assert len(_seg7_list) == 16


def get_seg_lines(n):
	seg_line_set = _seg7_list[n]
	return [line.value for line in seg_line_set]

if __name__ == '__main__':
	pass