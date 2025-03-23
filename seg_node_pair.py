from enum import Enum
from typing import Sequence, Iterator
from types import MappingProxyType
import numpy as np
from seg7yx import Seg7yx, Seg7yxSlant, Seg7Node6, node6
from seg_7_digits import Seg7Bit8, SEG7_DIGIT_ARRAY
from seg7bit8 import SEG7BIT8_ARRAY
NODE6_ARRAY = (
	(0, 0), (1, 0),
	(0, 1), (1, 1),
	(0, 2), (1, 2)
)
class Seg7Node6Pair:

	def __init__(self, pair: tuple[int, int]|tuple[int]):
		for x in pair:
			assert 0 <= pair[0] < 6
		self.pair = pair # (pair[0], pair[1]) if len(pair) > 1 else (pair[0],)
	def node6_map(self, nodes: node6):
		return (nodes[r] for r in self.pair)
	@classmethod
	def map_nodes(cls, nodes: node6, *spsp: 'Seg7Node6Pair')-> list[list[float]]:
		# assert len(node6) == 6
		#conv_tbl = [0, 1, 5, 2, 4, 3]
		rr = []
		for sp in spsp:
			rr += [nodes[r] for r in sp.pair]
		return rr

	@classmethod
	def map_node6_each(cls, node: node6, *spsp: 'Seg7Node6Pair')-> Iterator[list[Sequence[float]]]:
		# assert len(node6) == 6
		for j, sp in enumerate(spsp):
			xy_list = sp.node6_map(node)#[node6[r] for r in sp.pair]
			rr = []
			for i, xy in enumerate(xy_list):
				if i & 1:
					rr.append(xy)
					yield rr
					rr = []
				else:
					rr.append(xy)
			assert len(rr) == 0

	@classmethod
	def slant_scale_offset(cls, slant=Seg7yxSlant.SLANT02, scale=1, offset=(0, 0), *spsp: 'Seg7Node6Pair'):
		arr = np.array(slant.value) #seg7yx.Seg7yx(.to_list())
		if not isinstance(offset, np.ndarray):
			offset = np.array(offset)
		arr *= scale
		arr += offset
		xy6 = arr.tolist()
		assert len(xy6) == 6
		rr = []
		for sp in spsp:
			rr += [xy6[r] for r in sp.pair]
		return rr

class Seg7Elem(Enum):
	A = Seg7Node6Pair((0, 1))
	B = Seg7Node6Pair((1, 3))
	C = Seg7Node6Pair((3, 5))
	D = Seg7Node6Pair((5, 4))
	E = Seg7Node6Pair((4, 2))
	F = Seg7Node6Pair((2, 0))
	G = Seg7Node6Pair((2, 3))
	H = Seg7Node6Pair((5, )) # comma / period / dot

SEG7BIT8_TO_SEG7ELEM = MappingProxyType({
	Seg7Bit8.A: Seg7Elem.A,
	Seg7Bit8.B: Seg7Elem.B,
	Seg7Bit8.C: Seg7Elem.C,
	Seg7Bit8.D: Seg7Elem.D,
	Seg7Bit8.E: Seg7Elem.E,
	Seg7Bit8.F: Seg7Elem.F,
	Seg7Bit8.G: Seg7Elem.G,
	Seg7Bit8.H: Seg7Elem.H,
})

SEG7BIT8_TO_SEG7NODE6PAIR = MappingProxyType({
	Seg7Bit8.A: Seg7Elem.A.value,
	Seg7Bit8.B: Seg7Elem.B.value,
	Seg7Bit8.C: Seg7Elem.C.value,
	Seg7Bit8.D: Seg7Elem.D.value,
	Seg7Bit8.E: Seg7Elem.E.value,
	Seg7Bit8.F: Seg7Elem.F.value,
	Seg7Bit8.G: Seg7Elem.G.value,
	Seg7Bit8.H: Seg7Elem.H.value,
})

def expand_seg7bit8_to_seg7elems(s7: Seg7Bit8)-> list[Seg7Elem]:
	return [SEG7BIT8_TO_SEG7ELEM[b8] for b8 in SEG7BIT8_ARRAY if s7 & b8]

def encode_str_to_seg7bit8(n_s: str)-> Iterator[Seg7Bit8]:
	INDEX = '0123456789abcdef-'
	n_str = n_s + '\0'
	i = 0
	while i < len(n_str) - 1:
		ix = INDEX.index(n_str[i]) # << 1
		digital_bits: Seg7Bit8 = SEG7_DIGIT_ARRAY[ix]
		if n_str[i + 1] == '.':
			digital_bits |= Seg7Bit8.H
			i += 1
		yield digital_bits
		i += 1

def convert_str_to_seg7elems(n_s: str)-> Iterator[Sequence[Seg7Elem]]:
	from bin2 import Bin2
	INDEX = '0123456789abcdef-'
	n_str = n_s + '\0'
	bb = []
	i = 0
	while i < len(n_str) - 1:
		ix = INDEX.index(n_str[i]) # << 1
		digital_bits: Seg7Bit8 = SEG7_DIGIT_ARRAY[ix]
		if n_str[i + 1] == '.':
			digital_bits |= Seg7Bit8.H
			i += 1
		seg7elems = expand_seg7bit8_to_seg7elems(digital_bits)
		yield seg7elems
		i += 1

class SegmentStrokeFeeder:
	'''Feeds strokes from Seg7Bit8'''
	def __init__(self, scale: int=1, offset: Sequence[int]=(0, 0), slant=0.2, padding=(0.2, 0.2)):
		self.scale = scale
		self.offset = offset
		seg7node6 = Seg7Node6(slant=slant)
		self.slant_scale_offset_map = Seg7Node6.scale_offset(scale=scale, offset=offset, seg7node6=seg7node6)
		max_x = max(*[x for (x, y) in self.slant_scale_offset_map])
		max_y = max(*[y for (x, y) in self.slant_scale_offset_map])
		self.size = (round(max_x * (1 + padding[0])), round(max_y * (1 + padding[1])))
	def feeding(self, seg7bit8: Seg7Bit8):
		'''convert_seg7bit8_to_strokes_each'''
		for n, elem in enumerate(expand_seg7bit8_to_seg7elems(seg7bit8)):
			seg7node6pair = elem.value
			stroke: list[int] = []
			for i, xy in enumerate(seg7node6pair.node6_map(self.slant_scale_offset_map)):
				if i & 1:
					stroke.append(xy)
					yield np.array(stroke) # + tuple(xy)
					stroke = []
				else:
					stroke.append(xy) # = [round(x), round(y) for (x, y) in xy]
			if len(stroke) > 0:
				ofst = np.array(self.size) * 0.1
				strk0 = np.array(stroke[0]) + ofst
				dot_strk = np.array([0, ofst[0]]) / 2
				yield np.array([strk0, strk0 + dot_strk])
				#yield [round(r) for r in strk.ravel().tolist()]

if __name__ == '__main__':
	import sys
	from pprint import pp
	from ipycanvas import Canvas
	import seg7yx
	num = 3.12
	num_str = "%.2f" % num
	digits = list(encode_str_to_seg7bit8(num_str))
	scale = 20
	offset = [3, 5]
	stroke_feeder = SegmentStrokeFeeder(scale=scale, offset=offset, slant=0.2)
	from PIL import Image, ImageDraw
	img_box_size = [stroke_feeder.size[0] * len(digits), stroke_feeder.size[1]]
	img = Image.new('L', img_box_size, 0xff)
	drw = ImageDraw.Draw(img)
	drw.rectangle([0, 0, img_box_size[0] - 1, img_box_size[1] - 1],outline=0x55)
	x_shift = np.array([stroke_feeder.size[0],0])
	for n, digit in enumerate(digits):
		for i, strokes in enumerate(stroke_feeder.feeding(digit)):
			pp(strokes)
			shifted_strokes = strokes + n * x_shift
			_strokes = [round(s) for s in shifted_strokes.ravel().tolist()]
			drw.line(_strokes, width=i + 1)
	img.show()
	sys.exit(0)
	seg7node6 = Seg7Node6(slant=0.2)#.to_list()
	slant_scale_offset_map = Seg7Node6.scale_offset(scale=scale, offset=offset, seg7node6=seg7node6)
	max_x = max(*[x for (x, y) in slant_scale_offset_map])
	max_y = max(*[y for (x, y) in slant_scale_offset_map])
	size = (round(max_x * 1.2), round(max_y * 1.2))

	seg7elems = convert_str_to_seg7elems(num_str)

	for elems in seg7elems:
		img = Image.new('L', size, 0xff)
		drw = ImageDraw.Draw(img)
		for n, elem in enumerate(elems):
			seg7node6pair = elem.value
			stroke = []
			for i, xy in enumerate(seg7node6pair.node6_map(slant_scale_offset_map)):
				if i & 1:
					drw.line(stroke + [xy], fill=0, width=n + 1)
					stroke = []					
				else:
					stroke = [xy] #[x for x in xy]
			if len(stroke):
				ofst = np.array([5, 5])
				strk0 = np.array(stroke[0]) + ofst
				dot_strk = np.array([0, 5])
				strk = np.array([strk0, strk0 + dot_strk])
				stroke_list = strk.ravel().tolist()
				drw.line(stroke_list, fill=0, width=n + 1)					
		img.show()
	sys.exit(0)
	#canvas
'''	canvas = Canvas(width=200, height=200)
	canvas.stroke_style = "red"
	canvas.stroke_rect(round(offset[0] - scale * 0.2), round(offset[1]-scale * 0.2), round(scale * 1.6), round(scale * 2 * 1.4))
	canvas.stroke_style = "blue"
	canvas.line_width = 8
			#canvas.stroke_line(*stroke)
	pair = []
	for i, xy in enumerate(mapped_array):
		if i & 1:
			drw.line(pair + xy, 0)
		else:
			pair = xy
		def map_array(n):
		return slant_scale_offset_map[n]
			mapped_array = SegNodePair.map_node6(slant_scale_offset_map, *snp_array)
	mapped_array_ = [m for m in map(map_array, snp_array)]
	def map_func(sp: SegNodePair):
		return [slant_scale_offset_map[r] for r in sp.pair]
	slanted_snp_array = [m for m in map(map_func, snp_array)]'''

	# scaled_array = SegNodePair.slant_scale_offset(seg7yx.Seg7yxSlant.SLANT02, 10, (1, 2), *snp_array)

