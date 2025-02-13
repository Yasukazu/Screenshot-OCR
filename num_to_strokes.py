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

from strok7 import SegElem, _SEGELEM7DICT
_segelem_array: list[tuple[Seg7]] = [set()] * 16

_segpath_array = [] * 16
def load_segpath_array():
	'''call "slanted" for each element'''
	from seven_seg import load_7_seg_num_csv_as_df
	df = load_7_seg_num_csv_as_df()
	for i in range(16):
		dline = df[i: i + 1]
		elem_list = []
		for c in 'abcdefg':
			if int(list(dline[c])[0]) > 0:
				segpath = _SEGELEM7DICT[c]
				elem_list.append(segpath)
		_segpath_array[i] = elem_list
load_segpath_array()
def get_seg_list():
	return _segpath_array

from strok7 import SegLine
_seg7_array: list[set[Seg7]] = [set()] * 16
def load_seg7()-> list[set[SegLine]]:
	from seven_seg import load_7_seg_num_csv_as_df
	df = load_7_seg_num_csv_as_df()
	for i in range(16):
		dline = df[i: i + 1]
		seg_set = set()
		for c in 'abcdefg':
			if int(list(dline[c])[0]) > 0:
				seg = SegLine.get(c)
				seg_set.add(seg) # dlst.append(seg) # '1') # SegFlag.get(c).value)
		_seg7_array[i] = seg_set
	return _seg7_array
def get_seg7_list():
	return _seg7_array

def _load_seg7(pkl: str=SEVEN_SEG_STEM + PICKLE_EXT):
	rf = open(pkl, 'rb').read()
	csv_reader = pickle.loads(rf)
	for i, row in enumerate(csv_reader):
		seg = Seg7(*row)
		assert seg.h == i
		_seg7_array.append(seg)
	assert len(_seg7_array) == 16
	return _seg7_array

assert len(_seg7_array) == 16


def get_seg_lines(n):
	seg_line_set = _seg7_array[n]
	return [line.value for line in seg_line_set]

if __name__ == '__main__':
	pass