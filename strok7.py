from enum import Enum, StrEnum, Flag, auto
from typing import Sequence, Callable, TypeAlias
from functools import lru_cache
import csv
import numpy as np
from seg7yx import SlantedNode6

ii_ii: TypeAlias = tuple[tuple[int, int], tuple[int, int]]
i_i_tpl_tpl: TypeAlias = dict[str, ii_ii]

i_i_tpl: TypeAlias = tuple[int, int]
f_i_tpl: TypeAlias = tuple[float, int]
f_f_tpl: TypeAlias = tuple[float, float]

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
	X_MAX = 1 # 2 for dot
	Y_MAX = 2

	def __init__(self, x: int, y: int):
		assert 0 <= x <= Sp.X_MAX
		assert 0 <= x <= Sp.Y_MAX
		self.x = x
		self.y = y
		# self._slr = (1, 0.5, 0)[y]

	@property
	def _slr(self)-> float:
		return 1 - self.y / 2 # (1, 0.5, 0)[

	@property
	def xy(self)-> tuple[int, int]:
		return self.x, self.y

	def offset_self(self, offset=(0, 0)):
		self.x += offset[0]
		self.y += offset[1]

	def scale_self(self, n: int=1):
		self.x *= n
		self.y *= n

	def slanted_x(self, slant: StrokeSlant=NO_SLANT)-> float:
		return self.x + slant.value * self._slr

	#def slant_self(self, slant: StrokeSlant=NO_SLANT)-> None:		self.x = self.slant_x(slant)

	def slanted(self, slant: StrokeSlant=NO_SLANT)-> f_i_tpl:
		return self.slanted_x(slant), self.y

	def scale_offset(self, node6: SlantedNode6=SlantedNode6.SLANT02, scale: int=1, offset: tuple[int, int]=(0, 0))-> i_i_tpl:
		slant_map = node6.value
		slanted_x = slant_map[self.y][self.x]
		return round(scale * slanted_x) + offset[0], scale * self.y + offset[1]

class MySp(Sp):
	def __init__(self, x: int, y: int, slant_enum=StrokeSlant.SLANT00, scale_value=1, offset_value=(0, 0)):
		super().__init__(x, y)
		self.slant_enum = slant_enum
		self.scale_value = scale_value
		self.offset_value = offset_value
	
	def scale_offset(self, slant = StrokeSlant.SLANT00, scale = 1, offset = (0, 0)):
		return super().scale_offset(node6=self.slant_enum, scale=self.scale_value, offset=self.offset_value)

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

SEG_POINT_ARRAY = _S_A = (
	Sp(0, 0),
	Sp(1, 0),
	Sp(1, 1),
	Sp(1, 2),# comma
	Sp(0, 2),
	Sp(0, 1),
	)

class MySegPoints:
	'''Customized Seg Point array'''
	def __init__(self, slant: StrokeSlant=StrokeSlant.SLANT00, scale: int=1, offset: tuple[int, int]=(0, 0)):
		self.points = list(SEG_POINT_ARRAY)
		self.slant = slant
		self.scale = scale
		self.offset = offset

	def get(self, n: int):
		if not 0 <= n < len(SEG_POINT_ARRAY):
			raise ValueError(f"Out of range[0: {len(SEG_POINT_ARRAY)}]!")
		sp = self.points[n]
		if isinstance(sp, MySp):
			return sp
		sp = MySp(sp.x, sp.y, self.slant, self.scale, self.offset)
		self.points[n] = sp
		return sp

class SpPair(Enum):
	A = _S_A[0], _S_A[1]
	B = _S_A[1], _S_A[2]
	C = _S_A[2], _S_A[3]
	D = _S_A[3], _S_A[4]
	E = _S_A[4], _S_A[5]
	F = _S_A[5], _S_A[0]
	G = _S_A[5], _S_A[2] # minus / hyphen
	H = (_S_A[3],) # period / comma / dot
	#I = _6, _6 # dot
	@classmethod
	def expand_to_xy_list(cls, spp: 'SpPair')-> list[tuple[int, int]]:
		return [sp.xy for sp in spp.value]
	"""@classmethod
	def get(cls, c: str)-> 'SpPair':
		index = 'ABCDEFGHI'.index(c)
		return [cls.A, cls.B, cls.C, cls.D, cls.E, cls.F, cls.G, cls.H][index]

	@classmethod
	def extract(cls, c: str)-> list[tuple[float, int]]:
		return [sp.xy for sp in cls.get(c).value]"""


from functools import lru_cache
from PIL import ImageDraw
from strok7 import StrokeSlant
SEGPATH_SLANT = StrokeSlant.SLANT02
import numpy as np

class SegPath:
	def __init__(self, *spsp: Sp): # f_sp: Sp0, t_sp: Sp0):, max_cache=2
		self.path = np.array([sp.xy for sp in spsp]) #[sp for sp in spsp] # self.f = f_sp self.t = t_sp
		# self.slanted = lru_cache(maxsize=max_cache)(self._slanted)

	def get_path(self)-> list[int]:
		return self.path.ravel().tolist()

	def draw(self, drw: ImageDraw.ImageDraw, scale: int, offset: np.ndarray | tuple[int, int], line_width=1, fill=0):
		if type(offset) != np.typing.NDArray:
			offset = np.array(offset)
		path = scale * self.path + offset
		drw.line(path.ravel().tolist(), fill=fill, width=line_width)


	'''def _slanted(self, s=SEGPATH_SLANT, scale=1, offset=(0, 0)):
		return [pt.scale_offset(slant=s, scale=scale, offset=offset) for pt in self.path]'''
class CSegPath(SegPath):
	def draw(self, drw: ImageDraw.ImageDraw, scale=1, offset: np.ndarray | tuple[int, int]=(0, 0), line_width=1, fill=0):
		if type(offset) != np.typing.NDArray:
			offset = np.array(offset)
		path = scale * self.path + offset
		drw.circle(path.ravel().tolist(), radius=line_width, fill=fill)

class SegElem(Enum):
	A = SegPath(SEG_POINT_ARRAY[0], SEG_POINT_ARRAY[1])
	B = SegPath(SEG_POINT_ARRAY[1], SEG_POINT_ARRAY[2])
	C = SegPath(SEG_POINT_ARRAY[2], SEG_POINT_ARRAY[3])
	D = SegPath(SEG_POINT_ARRAY[3], SEG_POINT_ARRAY[4])
	E = SegPath(SEG_POINT_ARRAY[4], SEG_POINT_ARRAY[5])
	F = SegPath(SEG_POINT_ARRAY[5], SEG_POINT_ARRAY[0])
	G = SegPath(SEG_POINT_ARRAY[5], SEG_POINT_ARRAY[2])
	H = CSegPath(SEG_POINT_ARRAY[3]) # comma / period / dot


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
	def get(cls, c: str)-> 'SegLine':
		abcdefg = [cls.a, cls.b, cls.c, cls.d, cls.e, cls.f, cls.g]
		return abcdefg["abcdefg".index(c)]

class SegFlag(Flag): # same as Seg7
	G = 2
	F = 4
	E = 8
	D = 16
	C = 32
	B = 64
	A = 128
	
	@classmethod
	def get(cls, c: str, p=0):
		abcdefg = [cls.A, cls.B, cls.C, cls.D, cls.E, cls.F, cls.G]
		return abcdefg["ABCDEFG".index(c[p].upper())]
	
'''
def seg_flag_to_sp_pair(flag: SegFlag, seg_flag_dic={f: f.name for f in [
	SegFlag.A,
	SegFlag.B,
	SegFlag.C,
	SegFlag.D,
	SegFlag.E,
	SegFlag.F,
	SegFlag.G,
]})-> SpPair:
	pass
'''
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

def get_seg7_dict()-> dict[str, Sequence[tuple[int, int]]]:
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
	from pprint import pp
	from PIL import Image
	size = (100, 200)
	img = Image.new('L', size, 0xff)
	drw = ImageDraw.Draw(img)


	img.show()
	sys.exit(0)

	se = SegElem.H 	#se_a = SegElem.A
	sp = se.value
	sp.draw(drw=drw, scale=50, offset=(20, 20), line_width=4)
	
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