from enum import Enum
from typing import Sequence, Iterator
from types import MappingProxyType
import numpy as np
from seg7yx import Seg7yxSlant, Seg7Node6, node6
from seg_7_digits import Seg7Bit8, SEG7_ARRAY
from seg7bit8 import SEG7BIT8_ARRAY
class Seg7Node6Pair:
	def __init__(self, xyxy: Sequence[int]):
		assert 0 <= xyxy[0] < 6
		if len(xyxy) > 1:
			assert 0 <= xyxy[1] < 6
		self.pair = (xyxy[0], xyxy[1]) if len(xyxy) > 1 else (xyxy[0],)
	def node6_map(self, nodes: node6):
		return [nodes[r] for r in self.pair]
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
	B = Seg7Node6Pair((1, 2))
	C = Seg7Node6Pair((2, 3))
	D = Seg7Node6Pair((3, 4))
	E = Seg7Node6Pair((4, 5))
	F = Seg7Node6Pair((5, 0))
	G = Seg7Node6Pair((5, 2))
	H = Seg7Node6Pair((3, )) # comma / period / dot

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

def expand_seg7bit8_to_seg7elems(s7: Seg7Bit8)-> Sequence[Seg7Elem]:
	return tuple(SEG7BIT8_TO_SEG7ELEM[b8] for b8 in SEG7BIT8_ARRAY if s7 & b8)

def str_to_seg7elems(n_s: str)-> list[Sequence[Seg7Elem]]:
	from bin2 import Bin2
	INDEX = '0123456789abcdef-'
	n_str = n_s + '\0'
	bb = []
	i = 0
	while i < len(n_str) - 1:
		b = INDEX.index(n_str[i]) << 1
		if n_str[i + 1] == '.':
			b += 1
			i += 1
		b8 = Bin2(b).to_bit8()
		bb += [expand_seg7bit8_to_seg7elems(b8)]
		i += 1
	return bb
if __name__ == '__main__':
	import sys
	from pprint import pp
	import seg7yx
	num = 3.12 # int(sys.argv[1])
	num_str = "%.2f" % num
	seg7elems = str_to_seg7elems(num_str)
	b8s = SEG7_ARRAY[num]
	smsm = []
	for b8 in SEG7BIT8_ARRAY:
		if b8 & b8s:
			smsm.append(SEG7BIT8_TO_SEG7ELEM[b8].value)
	snp_array_list = [[s.value for s in sm] for sm in seg7elems] # [SegNodePairElem.B.value, SegNodePairElem.C.value]
	scale = 30
	offset = [10, 20]
	#slant02 = Seg7yxSlant.SLANT02.value #seg7yx.Seg7yx(seg7yx.).to_seg7()
	# slant02_array = np.array(slant02)
	seg7node6 = Seg7Node6(slant=0.2)
	slant_scale_offset_map = Seg7Node6.scale_offset(scale=scale, offset=offset, node6=seg7node6) #(slant02_array * scale + offset).tolist()
	max_x = max(*[x for (x, y) in slant_scale_offset_map])
	max_y = max(*[y for (x, y) in slant_scale_offset_map])
	size = (round(max_x * 1.2), round(max_y * 1.2))
	from ipycanvas import Canvas
	canvas = Canvas(width=200, height=200)
	canvas.stroke_style = "blue"
	canvas.stroke_rect(round(offset[0] - scale * 0.2), round(offset[1]-scale * 0.2), round(scale * 1.6), round(scale * 2 * 1.4))
	canvas.stroke_style = "blue"
	canvas.line_width = 8
	canvas.line_join = 'bevel'
	for xy_list in Seg7Node6Pair.map_node6_each(slant_scale_offset_map, *snp_array):
		pp(xy_list)
		canvas.stroke_line(*xy_list)
	canvas
	sys.exit(0)
	from PIL import Image, ImageDraw
	img = Image.new('L', size, 0xff)
	drw = ImageDraw.Draw(img)
	pair = []
	for i, xy in enumerate(mapped_array):
		if i & 1:
			drw.line(pair + xy, 0)
		else:
			pair = xy
	img.show()
	'''
		def map_array(n):
		return slant_scale_offset_map[n]
			mapped_array = SegNodePair.map_node6(slant_scale_offset_map, *snp_array)
	mapped_array_ = [m for m in map(map_array, snp_array)]
	def map_func(sp: SegNodePair):
		return [slant_scale_offset_map[r] for r in sp.pair]
	slanted_snp_array = [m for m in map(map_func, snp_array)]'''

	# scaled_array = SegNodePair.slant_scale_offset(seg7yx.Seg7yxSlant.SLANT02, 10, (1, 2), *snp_array)

