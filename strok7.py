from enum import Enum, StrEnum, Flag, auto
from typing import Sequence
import csv

type ii_ii = tuple[tuple[int, int], tuple[int, int]]
type i_i_tpl_tpl = dict[str, ii_ii]

type i_i_tpl = tuple[int, int]
type f_i_tpl = tuple[float, int]
type f_f_tpl = tuple[float, float]

class StrokeSlant(Enum):
	SLANT00 = 0
	SLANT02 = 0.2

NO_SLANT = StrokeSlant.SLANT00
STANDARD_SLANT = StrokeSlant.SLANT02
class Sp:
	'''
	slant p0 p1
	slant/2 p5 p2
	p4 p3
	'''
	def __init__(self, x: int, y: int):
		assert x in (0, 1, 2) # 2 for comma
		assert y in (0, 1, 2)
		self.x = x
		self.y = y
		self._slr = (1, 0.5, 0)[y]

	@property
	def xy(self)-> tuple[int, int]:
		return self.x, self.y

	def offset(self, offset=(0, 0)):
		self.x += offset[0]
		self.y += offset[1]

	def scale(self, n: int=1):
		self.x *= n
		self.y *= n

	def slant_x(self, slant: StrokeSlant=NO_SLANT)-> float:
		return self.x + slant.value * self._slr

	def slant(self, slant: StrokeSlant=NO_SLANT)-> f_i_tpl:
		self.x = self.slant_x(slant)

	def slanted(self, slant: StrokeSlant=NO_SLANT)-> f_i_tpl:
		return self.slant_x(slant), self.y

	def scale_offset(self, slant: StrokeSlant=StrokeSlant.SLANT00, scale: int=1, offset: tuple[int, int]=(0, 0))-> i_i_tpl:
		'''also slant'''
		slanted_x = self.slant_x(slant) # / (1 + slant.value)
		return round(scale * slanted_x) + offset[0], scale * self.y + offset[1]

class Sp0(Sp):
	def __init__(self, x: int):
		super().__init__(x=x, y=0)

class Sp1(Sp):
	def __init__(self, x: int):
		super().__init__(x=x, y=1)

class Sp2(Sp):
	def __init__(self, x: int):
		super().__init__(x=x, y=2)

abcdef_seg = (
	(0, 0),
	(1, 0),
	(1, 1),
	(1, 2),
	(0, 2),
	(0, 1),
	(0, 0),
	)

class SpPair(Enum):
	_0 = Sp(0, 0)
	_1 = Sp(1, 0)
	_2 = Sp(1, 1)
	_3 = Sp(1, 2)
	_4 = Sp(0, 2)
	_5 = Sp(0, 1)
	_6 = Sp(2, 2) # comma / period / dot

	A = _0, _1
	B = _1, _2
	C = _2, _3
	D = _3, _4
	E = _4, _5
	F = _5, _0
	G = _5, _2 # minus / hyphen
	H = _1, _4 # per / slash
	I = _6, _6 # dot

	@classmethod
	def get(cls, c: str)-> 'SpPair':
		index = 'ABCDEFGHI'.index(c)
		return [cls.A, cls.B, cls.C, cls.D, cls.E, cls.F, cls.G, cls.H, cls.I][index]
	@classmethod
	def extract(cls, c: str)-> list[tuple[int, int]]:
		return [sp.xy for sp in cls.get(c).value]


class Seg7Path(Enum):
	a = SpPair._0, SpPair._1

SEG_POINT_ARRAY = (
	Sp(0, 0),
	Sp(1, 0),
	Sp(1, 1),
	Sp(1, 2),
	Sp(0, 2),
	Sp(0, 1),
	)

from functools import lru_cache
SEGPATH_SLANT = 0.2
class SegPath:
	def __init__(self, *spsp: Sequence[Sp], max_cache=2): # f_sp: Sp0, t_sp: Sp0):
		self.path: Sequence[Sp] = spsp # self.f = f_sp self.t = t_sp
		self.slanted = lru_cache(maxsize=max_cache)(self._slanted)

	def get_path(self):
		return list(self.path)

	def _slanted(self, s=SEGPATH_SLANT, scale=1.0, offset=(0,0)):
		return [pt.slant(s=s, scale=scale, offset=offset) for pt in self.path]


class SegElem(Enum):
	A = SegPath(SEG_POINT_ARRAY[0], SEG_POINT_ARRAY[1])
	B = SegPath(SEG_POINT_ARRAY[1], SEG_POINT_ARRAY[2])
	C = SegPath(SEG_POINT_ARRAY[2], SEG_POINT_ARRAY[3])
	D = SegPath(SEG_POINT_ARRAY[3], SEG_POINT_ARRAY[4])
	E = SegPath(SEG_POINT_ARRAY[4], SEG_POINT_ARRAY[5])
	F = SegPath(SEG_POINT_ARRAY[5], SEG_POINT_ARRAY[0])
	G = SegPath(SEG_POINT_ARRAY[5], SEG_POINT_ARRAY[2])
	H = SegPath(SEG_POINT_ARRAY[1], SEG_POINT_ARRAY[4]) # comma


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
def get_segpath_for_c(c: str, segelem_dict={e.name.lower(): e.value for e in SEGELEMS})-> SegPath:
	_c = c.lower()[0]
	seg_names = 'abcdefgh'
	if _c not in seg_names:
		raise ValueError(f"'{c[0]}' needs be in {seg_names}!")
	return segelem_dict[_c]
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

def print_seg_point_enum():
	print("class SegPoints(Enum):")
	for n, points in abcdef_seg:
		print(f"\t_{n} = {str(points)}")

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