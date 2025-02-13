from enum import Enum, StrEnum, Flag, auto
import csv

type ii_ii = tuple[tuple[int, int], tuple[int, int]]
type i_i_tpl_tpl = dict[str, ii_ii]


class SegName(StrEnum):
	A = auto()
	B = auto()
	C = auto()
	D = auto()
	E = auto()
	F = auto()
	G = auto()

class Sp0:
	Y = 0
	def __init__(self, x: int):
		self.x = x
	def slant(self, s=0):
		return self.x + s, self.Y
class Sp1(Sp0):
	Y = 1
	def slant(self, s=0):
		return (s / 2.0) + self.x, self.Y
class Sp2(Sp0):
	Y = 2
	def slant(self, s=0):
		return self.x, self.Y

abcdef_seg = ((0, 0),
	(1, 0),
	(1, 1),
	(1, 2),
	(0, 2),
	(0, 1),
	(0, 0))

SEG_POINTS = [Sp0(0),
	Sp0(1),
	Sp1(1),
	Sp2(1),
	Sp2(0),
	Sp1(0),
	Sp0(0)]

class SegPath:
	def __init__(self, *spsp): # f_sp: Sp0, t_sp: Sp0):
		self.path = spsp # self.f = f_sp self.t = t_sp
	def get_path(self):
		for pt in self.path:
			yield pt
		# return self.f, self.t
	def slanted(self, s=0.0):
		for pt in self.path:
			yield pt.slant(s) # self.f.slant(), self.t.slant()

class SegElem(Enum):
	A = SegPath(SEG_POINTS[0], SEG_POINTS[1])
	B = SegPath(SEG_POINTS[1], SEG_POINTS[2])
	C = SegPath(SEG_POINTS[2], SEG_POINTS[3])
	D = SegPath(SEG_POINTS[3], SEG_POINTS[4])
	E = SegPath(SEG_POINTS[4], SEG_POINTS[5])
	F = SegPath(SEG_POINTS[5], SEG_POINTS[6])
	G = SegPath(Sp1(0), Sp1(1))

SEGELEM7 = (
        SegElem.A,
        SegElem.B,
        SegElem.C,
        SegElem.D,
        SegElem.E,
        SegElem.F,
        SegElem.G,
			)
_SEGELEM7DICT = {e.name.lower(): e.value for e in SEGELEM7}
def get_segelem7dict():
	return _SEGELEM7DICT
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
	if len(sys.argv) == 1:
		print("Needs funcspec.")
		sys.exit(1)
	match sys.argv[1]:
		case 'save_strk_dict':
			save_strk_dict()
		case 'save_np_strk_dict':
			save_np_strk_dict()
