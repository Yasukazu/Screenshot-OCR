from enum import Enum
import csv
abcdef_seg = ((0, 0),
	(1, 0),
	(1, 1),
	(1, 2),
	(0, 2),
	(0, 1),
	(0, 0))

class SegLine(Enum):
	a = (abcdef_seg[0], abcdef_seg[1])
	b = (abcdef_seg[1], abcdef_seg[2])
	c = (abcdef_seg[2], abcdef_seg[3])
	d = (abcdef_seg[3], abcdef_seg[4])
	e = (abcdef_seg[4], abcdef_seg[5])
	f = (abcdef_seg[5], abcdef_seg[6])
	g = ((0, 1), (1, 1))
	
	@classmethod
	def get(cls, c: str):
		abcdefg = [cls.a, cls.b, cls.c, cls.d, cls.e, cls.f, cls.g]
		return abcdefg["abcdefg".index(c)]

class Segment7:
	dic = {c: (abcdef_seg[i], abcdef_seg[i + 1]) for i, c in enumerate('abcdef')}
	dic['g'] = ((0, 1), (1, 1))
	def __init__(self, f, t):
		self.f = int(f)
		self.t = int(t)
	def export(self)-> tuple[int, int]:
		return self.f, self.t



strk7: list[Segment7] = []

with open('abcdef-7.csv', encoding='utf8') as csv_file:
	csv_reader = csv.reader(csv_file)
	for row in csv_reader:
		seg = Segment7(*row)
		strk7.append(seg)

type i_i_tpl_tpl = dict[str, tuple[tuple[int, int], tuple[int, int]]]
strk_dic: i_i_tpl_tpl = {}

for i, j in enumerate('abcdef'):
	strk_dic[j] = (strk7[i].export(), strk7[i + 1].export())

strk_dic['g'] = ((0, 1), (1, 1))

def get_seg7_dict()-> dict[str, i_i_tpl_tpl]:
	return {c: SegLine.get(c).value for c in 'abcdefg'}

def get_strk_dict()-> i_i_tpl_tpl:
	return Segment7.dic
	'''strk_dic: i_i_tpl_tpl = {}
	for i, j in enumerate('abcdef'):
		strk_dic[j] = (strk7[i].export(), strk7[i + 1].export())
	strk_dic['g'] = ((0, 1), (1, 1))'''

def get_np_strk_dict():
	return {k: np.array(v, int) for (k, v) in Segment7.dic.items() }

STRK_DICT_STEM = 'strk_dict'
PICKLE_EXT = '.pkl'

def save_strk_dict(filename: str=STRK_DICT_STEM + PICKLE_EXT):
	strk_dic = get_strk_dict()
	with open(filename, 'wb') as wf:
		pickle.dump(strk_dic, wf)

import numpy as np

NP_STRK_DICT_PKL = 'np_strk_dict.pkl'

import pickle

def save_np_strk_dict():
	np_strk_dict = {k: np.array(v, int) for (k, v) in strk_dic.items() }
	with open(NP_STRK_DICT_PKL, 'wb') as wf:
		pickle.dump(np_strk_dict, wf)

if __name__ == '__main__':
	import sys
	if len(sys.argv) == 1:
		print("Needs funcspec.")
		sys.exit(1)
	match sys.argv[1]:
		case 'save_strk_dict':
			save_strk_dict()
		case 'save_np_strk_dict':
			save_np_strk_dict()
