from enum import Enum, StrEnum, Flag, auto
from typing import Sequence
import csv

type ii_ii = tuple[tuple[int, int], tuple[int, int]]
type i_i_tpl_tpl = dict[str, ii_ii]

type f_i_tpl = tuple[float, int]

STANDARD_SLANT = 0.2
class Sp0:
	def __init__(self, x: int):
		self.x: int = x
	def slant(self, s=STANDARD_SLANT)-> f_i_tpl:
		return s * self._slr + self.x, self.y
	@property
	def _slr(self):
		return 1
	@property
	def xy(self)-> tuple[int, int]:
		return self.x, self.y
	@property
	def y(self)-> int:
		return 0
class Sp1(Sp0):
	@property
	def _slr(self):
		return 0.5
	@property
	def y(self)-> int:
		return 1
class Sp2(Sp0):
	@property
	def _slr(self):
		return 0
	@property
	def y(self)-> int:
		return 2

abcdef_seg = (
	(0, 0),
	(1, 0),
	(1, 1),
	(1, 2),
	(0, 2),
	(0, 1),
	(0, 0),
	)


SEG_POINTS = (
	Sp0(0),
	Sp0(1),
	Sp1(1),
	Sp2(1),
	Sp2(0),
	Sp1(0),
	)

from functools import lru_cache
SEGPATH_SLANT = 0.2
class SegPath:
	def __init__(self, *spsp: Sequence[Sp0], max_cache=2): # f_sp: Sp0, t_sp: Sp0):
		self.path: Sequence[Sp0] = spsp # self.f = f_sp self.t = t_sp
		self.slanted = lru_cache(maxsize=max_cache)(self._slanted)

	def get_path(self):
		return list(self.path)

	def _slanted(self, s=SEGPATH_SLANT):
		return [pt.slant(s) for pt in self.path]


class SegElem(Enum):
	A = SegPath(SEG_POINTS[0], SEG_POINTS[1])
	B = SegPath(SEG_POINTS[1], SEG_POINTS[2])
	C = SegPath(SEG_POINTS[2], SEG_POINTS[3])
	D = SegPath(SEG_POINTS[3], SEG_POINTS[4])
	E = SegPath(SEG_POINTS[4], SEG_POINTS[5])
	F = SegPath(SEG_POINTS[5], SEG_POINTS[0])
	G = SegPath(SEG_POINTS[5], SEG_POINTS[2])
	H = SegPath(SEG_POINTS[1], SEG_POINTS[4]) # comma



SEGELEMS = (
        SegElem.A,
        SegElem.B,
        SegElem.C,
        SegElem.D,
        SegElem.E,
        SegElem.F,
        SegElem.G,
        SegElem.H,
			)
def get_segelem_dict(segelem_dict={e.name.lower(): e.value for e in SEGELEMS}):
	return segelem_dict
def get_segpath_dict(segelem_dict=get_segelem_dict()):
	return {k: v.slanted() for (k,v) in segelem_dict.items()}
class SegLine(Enum):
	a = (abcdef_seg[0], abcdef_seg[1])
	b = (abcdef_seg[1], abcdef_seg[2])
	c = (abcdef_seg[2], abcdef_seg[3])
	d = (abcdef_seg[3], abcdef_seg[4])
	e = (abcdef_seg[4], abcdef_seg[5])
	f = (abcdef_seg[5], abcdef_seg[0])
	g = ((0, 1), (1, 1))
	
	@classmethod
	def get(cls, c: str):
		abcdefg = [cls.a, cls.b, cls.c, cls.d, cls.e, cls.f, cls.g]
		return abcdefg["abcdefg".index(c)]

class SegFlag(Flag):
	a = auto()
	b = auto()
	c = auto()
	d = auto()
	e = auto()
	f = auto()
	g = auto()
	
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
	dic = get_segelem_dict()
	for c in 'abcdefg':
		a_path = dic[c].slanted()
		print(f"{c}: {a_path}")
	sys.exit(0)
	if len(sys.argv) == 1:
		print("Needs funcspec.")
		sys.exit(1)
	match sys.argv[1]:
		case 'save_strk_dict':
			save_strk_dict()
		case 'save_np_strk_dict':
			save_np_strk_dict()
