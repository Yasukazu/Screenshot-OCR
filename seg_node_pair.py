from enum import Enum
from typing import Sequence
import numpy as np
from seg7yx import Seg7yxSlant
from seg_7_digits import Bit8, SEG7_ARRAY, BIT8_ARRAY
class SegNodePair:
	def __init__(self, xyxy: Sequence[int]):
		assert 0 <= xyxy[0] < 6
		if len(xyxy) > 1:
			assert 0 <= xyxy[1] < 6
		self.pair = (xyxy[0], xyxy[1]) if len(xyxy) > 1 else (xyxy[0],)

	@classmethod
	def map_node6(cls, node6: Sequence[Sequence[float]], *spsp: 'SegNodePair')-> list[list[float]]:
		assert len(node6) == 6
		#conv_tbl = [0, 1, 5, 2, 4, 3]
		rr = []
		for sp in spsp:
			rr += [node6[r] for r in sp.pair]
		return rr

	@classmethod
	def slant_scale_offset(cls, slant=Seg7yxSlant.SLANT02, scale=1, offset=(0, 0), *spsp: 'SegNodePair'):
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

class SegNodePairElem(Enum):
	A = SegNodePair((0, 1))
	B = SegNodePair((1, 2))
	C = SegNodePair((2, 3))
	D = SegNodePair((3, 4))
	E = SegNodePair((4, 5))
	F = SegNodePair((5, 0))
	G = SegNodePair((5, 2))
	H = SegNodePair((3, )) # comma / period / dot

BIT8_TO_SEG_NODE_PAIR_ELEM = {
	Bit8.A: SegNodePairElem.A,
	Bit8.B: SegNodePairElem.B,
	Bit8.C: SegNodePairElem.C,
	Bit8.D: SegNodePairElem.D,
	Bit8.E: SegNodePairElem.E,
	Bit8.F: SegNodePairElem.F,
	Bit8.G: SegNodePairElem.G,
	Bit8.H: SegNodePairElem.H,
}

if __name__ == '__main__':
	import sys
	from pprint import pp
	import seg7yx
	num = int(sys.argv[1])
	b8s = SEG7_ARRAY[num]
	smsm = []
	for b8 in BIT8_ARRAY:
		if b8 & b8s:
			smsm.append(BIT8_TO_SEG_NODE_PAIR_ELEM[b8].value)
	snp_array = smsm # [SegNodePairElem.B.value, SegNodePairElem.C.value]
	scale = 30
	offset = np.array([10, 20])
	slant02 = Seg7yxSlant.SLANT02.value #seg7yx.Seg7yx(seg7yx.).to_seg7()
	slant02_array = np.array(slant02)
	slant_scale_offset_map = (slant02_array * scale + offset).tolist()
	mapped_array = SegNodePair.map_node6(slant_scale_offset_map, *snp_array)
	from PIL import Image, ImageDraw
	max_x = max(*[x for (x, y) in mapped_array])
	max_y = max(*[y for (x, y) in mapped_array])
	size = (round(max_x * 1.2), round(max_y * 1.2))
	img = Image.new('L', size, 0xff)
	drw = ImageDraw.Draw(img)
	pair = []
	for i, xy in enumerate(mapped_array):
		if i & 1:
			drw.line(pair + xy, 0)
		else:
			pair = xy
	img.show()
	'''def map_func(sp: SegNodePair):
		return [slant_scale_offset_map[r] for r in sp.pair]
	slanted_snp_array = [m for m in map(map_func, snp_array)]'''

	# scaled_array = SegNodePair.slant_scale_offset(seg7yx.Seg7yxSlant.SLANT02, 10, (1, 2), *snp_array)

