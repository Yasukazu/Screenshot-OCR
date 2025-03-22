from typing import Sequence, Callable
from types import MappingProxyType
from enum import Flag, IntEnum
from strok7 import SpPair, SegElem
from seg7bit8 import Seg7Bit8

seg_7_digits: Sequence[int] = (
	# abcdefgh
	0b11111100, # 00:252
	0b01100000, # 01:96
	0b11011010, # 02:218
	0b11110010, # 03:242
	0b01100110, # 04:102
	0b10110110, # 05:182
	0b10111110, # 06:190
	0b11100000, # 07:224
	0b11111110, # 08:254
	0b11100110, # 09:230
	0b11101110, # 0A:238
	0b00111110, # 0B:62
	0b00011010, # 0C:26
	0b01111010, # 0D:122
	0b10011110, # 0E:158
	0b10001110, # 0F:142
	0b00000010, # 10:2
)

from seg7bit8 import SEG7BIT8_ARRAY

SEG7BIT8_TO_SP_PAIR = MappingProxyType({
	Seg7Bit8.A: SpPair.A,
	Seg7Bit8.B: SpPair.B,
	Seg7Bit8.C: SpPair.C,
	Seg7Bit8.D: SpPair.D,
	Seg7Bit8.E: SpPair.E,
	Seg7Bit8.F: SpPair.F,
	Seg7Bit8.G: SpPair.G,
	Seg7Bit8.H: SpPair.H, # dot
})

SEG7BIT8_VALUE_TO_SP_PAIR = {
	Seg7Bit8.A.value: SpPair.A,
	Seg7Bit8.B.value: SpPair.B,
	Seg7Bit8.C.value: SpPair.C,
	Seg7Bit8.D.value: SpPair.D,
	Seg7Bit8.E.value: SpPair.E,
	Seg7Bit8.F.value: SpPair.F,
	Seg7Bit8.G.value: SpPair.G,
	Seg7Bit8.H.value: SpPair.H,
}

SEG7BIT8_VALUE_TO_SEG_ELEM = {
	Seg7Bit8.A.value: SegElem.A,
	Seg7Bit8.B.value: SegElem.B,
	Seg7Bit8.C.value: SegElem.C,
	Seg7Bit8.D.value: SegElem.D,
	Seg7Bit8.E.value: SegElem.E,
	Seg7Bit8.F.value: SegElem.F,
	Seg7Bit8.G.value: SegElem.G,
	Seg7Bit8.H.value: SegElem.H,
}

SEG7BIT8_TO_SEG_ELEM = {
	Seg7Bit8.A: SegElem.A,
	Seg7Bit8.B: SegElem.B,
	Seg7Bit8.C: SegElem.C,
	Seg7Bit8.D: SegElem.D,
	Seg7Bit8.E: SegElem.E,
	Seg7Bit8.F: SegElem.F,
	Seg7Bit8.G: SegElem.G,
	Seg7Bit8.H: SegElem.H,
}

def str_to_seg_elems(n_s: str)-> list[Sequence[SegElem]]:
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
		bb += [expand_bit8_to_seg_elems(b8)]
		i += 1
	return bb

def expand_bin_to_seg_elems(bn: int)-> Sequence[SegElem]:
	return tuple(SEG7BIT8_VALUE_TO_SEG_ELEM[bit] for bit in (Seg7Bit8.A.value, Seg7Bit8.B.value, Seg7Bit8.C.value, Seg7Bit8.D.value, Seg7Bit8.E.value, Seg7Bit8.F.value, Seg7Bit8.G.value, Seg7Bit8.H.value) if bit & bn)

def expand_bit8_to_seg_elems(bits: Seg7Bit8)-> Sequence[SegElem]:
	# bits = bin2_to_bit8(bn)
	return tuple(SEG7BIT8_TO_SEG_ELEM[bit] for bit in SEG7BIT8_ARRAY if bit & bits) # (Seg7Bit8.A, Seg7Bit8.B, Seg7Bit8.C, Seg7Bit8.D, Seg7Bit8.E, Seg7Bit8.F, Seg7Bit8.G, Seg7Bit8.H)

def expand_bin2_to_seg_elems(bn: int)-> Sequence[SegElem]:
	bits = bin2_to_bit8(Bin2(bn))
	return tuple(SEG7BIT8_VALUE_TO_SEG_ELEM[bit] for bit in (Seg7Bit8.A, Seg7Bit8.B, Seg7Bit8.C, Seg7Bit8.D, Seg7Bit8.E, Seg7Bit8.F, Seg7Bit8.G, Seg7Bit8.H) if bit & bits)

def expand_bin_to_sp_pairs(bn: int)-> Sequence[SpPair]:
	return tuple(SEG7BIT8_TO_SP_PAIR[bit] for bit in (Seg7Bit8.A.value, Seg7Bit8.B.value, Seg7Bit8.C.value, Seg7Bit8.D.value, Seg7Bit8.E.value, Seg7Bit8.F.value, Seg7Bit8.G.value, Seg7Bit8.H.value) if bit & bn)

def expand_bin_to_xy_list_list(bn: int)-> list[list[tuple[int, int]]]:
	return [SpPair.expand_to_xy_list(spp) for spp in expand_bin_to_sp_pairs(bn)]

def expand_to_sp_pairs(seg7: Seg7Bit8)-> Sequence[SpPair]:
	return tuple(SEG7BIT8_TO_SP_PAIR[seg] for seg in (Seg7Bit8.A, Seg7Bit8.B, Seg7Bit8.C, Seg7Bit8.D, Seg7Bit8.E, Seg7Bit8.F, Seg7Bit8.G) if seg7 & seg)

def expand_to_xy_list_list(seg7: Seg7Bit8)-> list[list[tuple[int, int]]]:
	return [SpPair.expand_to_xy_list(spp) for spp in expand_to_sp_pairs(seg7)]

def c_to_seg_7(c: str, C_TO_SEG7 = {
		'a': Seg7Bit8.A,
		'b': Seg7Bit8.B,
		'c': Seg7Bit8.C,
		'd': Seg7Bit8.D,
		'e': Seg7Bit8.E,
		'f': Seg7Bit8.F,
		'g': Seg7Bit8.G,
})-> Callable[[str], Seg7Bit8]:
	return C_TO_SEG7[c]

SEG7_DIGIT_ARRAY = (
	(Seg7Bit8.A | Seg7Bit8.B | Seg7Bit8.C | Seg7Bit8.D | Seg7Bit8.E | Seg7Bit8.F),
	(Seg7Bit8.B | Seg7Bit8.C),
	(Seg7Bit8.A | Seg7Bit8.B | Seg7Bit8.D | Seg7Bit8.E | Seg7Bit8.G),
	(Seg7Bit8.A | Seg7Bit8.B | Seg7Bit8.C | Seg7Bit8.D | Seg7Bit8.G),
	(Seg7Bit8.B | Seg7Bit8.C | Seg7Bit8.F | Seg7Bit8.G),
	(Seg7Bit8.A | Seg7Bit8.C | Seg7Bit8.D | Seg7Bit8.F | Seg7Bit8.G),
	(Seg7Bit8.A | Seg7Bit8.C | Seg7Bit8.D | Seg7Bit8.E | Seg7Bit8.F | Seg7Bit8.G),
	(Seg7Bit8.A | Seg7Bit8.B | Seg7Bit8.C),
	(Seg7Bit8.A | Seg7Bit8.B | Seg7Bit8.C | Seg7Bit8.D | Seg7Bit8.E | Seg7Bit8.F | Seg7Bit8.G),
	(Seg7Bit8.A | Seg7Bit8.B | Seg7Bit8.C | Seg7Bit8.F | Seg7Bit8.G),
	(Seg7Bit8.A | Seg7Bit8.B | Seg7Bit8.C | Seg7Bit8.E | Seg7Bit8.F | Seg7Bit8.G),
	(Seg7Bit8.C | Seg7Bit8.D | Seg7Bit8.E | Seg7Bit8.F | Seg7Bit8.G),
	(Seg7Bit8.D | Seg7Bit8.E | Seg7Bit8.G),
	(Seg7Bit8.B | Seg7Bit8.C | Seg7Bit8.D | Seg7Bit8.E | Seg7Bit8.G),
	(Seg7Bit8.A | Seg7Bit8.D | Seg7Bit8.E | Seg7Bit8.F | Seg7Bit8.G),
	(Seg7Bit8.A | Seg7Bit8.E | Seg7Bit8.F | Seg7Bit8.G),
	(Seg7Bit8.G),
	Seg7Bit8.H
)

def hex_to_bit8(n: int)-> Seg7Bit8:
	'''16 for hyphen/minus, 17 for comma/period'''
	if not (0 <= n < len(SEG7_DIGIT_ARRAY)):
		raise ValueError("Out of hexadecimal range!")
	return SEG7_DIGIT_ARRAY[n]


def bin_to_bit8(b: int)-> Seg7Bit8:
	if not 0 < b < 256:
		raise ValueError("Needs non-nul byte!")
	seg7_list = [seg for seg in (Seg7Bit8.A, Seg7Bit8.B, Seg7Bit8.C, Seg7Bit8.D, Seg7Bit8.E, Seg7Bit8.F, Seg7Bit8.G, Seg7Bit8.H) if b & seg.value]
	if len(seg7_list) == 1:
		return seg7_list[0]
	s_0 = seg7_list[0]
	after_0 = seg7_list[1:]
	for s in after_0:
		s_0 |= s
	return s_0


SEG_POINT_PAIR_DIGIT_ARRAY = (
	(SpPair.A, SpPair.B, SpPair.C, SpPair.D, SpPair.E, SpPair.F),
	(SpPair.B, SpPair.C),
	(SpPair.A, SpPair.B, SpPair.D, SpPair.E, SpPair.G),
	(SpPair.A, SpPair.B, SpPair.C, SpPair.D, SpPair.G),
	(SpPair.B, SpPair.C, SpPair.F, SpPair.G),
	(SpPair.A, SpPair.C, SpPair.D, SpPair.F, SpPair.G),
	(SpPair.A, SpPair.C, SpPair.D, SpPair.E, SpPair.F, SpPair.G),
	(SpPair.A, SpPair.B, SpPair.C),
	(SpPair.A, SpPair.B, SpPair.C, SpPair.D, SpPair.E, SpPair.F, SpPair.G),
	(SpPair.A, SpPair.B, SpPair.C, SpPair.F, SpPair.G),
	(SpPair.A, SpPair.B, SpPair.C, SpPair.E, SpPair.F, SpPair.G),
	(SpPair.C, SpPair.D, SpPair.E, SpPair.F, SpPair.G),
	(SpPair.D, SpPair.E, SpPair.G),
	(SpPair.B, SpPair.C, SpPair.D, SpPair.E, SpPair.G),
	(SpPair.A, SpPair.D, SpPair.E, SpPair.F, SpPair.G),
	(SpPair.A, SpPair.E, SpPair.F, SpPair.G),
	(SpPair.G,)
)

def digit_to_sp_pair(n: int)-> Sequence[SpPair]:
	if not (0 <= n < len(SEG_POINT_PAIR_DIGIT_ARRAY)):
		raise ValueError("Out of digit range!")
	return SEG_POINT_PAIR_DIGIT_ARRAY[n]

seg_7_array: Sequence[Sequence[str]] = (
		('a', 'b', 'c', 'd', 'e', 'f'),
		('b', 'c'),
		('a', 'b', 'd', 'e', 'g'),
		('a', 'b', 'c', 'd', 'g'),
		('b', 'c', 'f', 'g'),
		('a', 'c', 'd', 'f', 'g'),
		('a', 'c', 'd', 'e', 'f', 'g'),
		('a', 'b', 'c'),
		('a', 'b', 'c', 'd', 'e', 'f', 'g'),
		('a', 'b', 'c', 'f', 'g'),
		('a', 'b', 'c', 'e', 'f', 'g'),
		('c', 'd', 'e', 'f', 'g'),
		('d', 'e', 'g'),
		('b', 'c', 'd', 'e', 'g'),
		('a', 'd', 'e', 'f', 'g'),
		('a', 'e', 'f', 'g'),
		('g',),
)

type seg_7_tuple = tuple[str,str,str,str,str,str,str]

homo_seg_7_array: Sequence[seg_7_tuple] = (
		('a', 'b', 'c', 'd', 'e', 'f', ''), # 0
		('', 'b', 'c', '', '', '', ''), # 1
		('a', 'b', '', 'd', 'e', '', 'g'), # 2
		('a', 'b', 'c', 'd', '', '', 'g'), # 3
		('', 'b', 'c', '', '', 'f', 'g'), # 4
		('a', '', 'c', 'd', '', 'f', 'g'), # 5
		('a', '', 'c', 'd', 'e', 'f', 'g'), # 6
		('a', 'b', 'c', '', '', '', ''), # 7
		('a', 'b', 'c', 'd', 'e', 'f', 'g'), # 8
		('a', 'b', 'c', '', '', 'f', 'g'), # 9
		('a', 'b', 'c', '', 'e', 'f', 'g'), # A
		('', '', 'c', 'd', 'e', 'f', 'g'), # B
		('', '', '', 'd', 'e', '', 'g'), # C
		('', 'b', 'c', 'd', 'e', '', 'g'), # D
		('a', '', '', 'd', 'e', 'f', 'g'), # E
		('a', '', '', '', 'e', 'f', 'g'), # F
		('', '', '', '', '', '', 'g'), # 10
)
def get_seg_7_set(set_list=[], digits_list=seg_7_digits)-> list[set[str]]:
	if len(set_list) == 0:
		for i in digits_list:
			st = set()
			n = i >> 1
			for j, c in enumerate(reversed('abcdefg')):
				bp = 1 << j
				if n & bp:
					st.add(c)
			set_list.append(st)
	return set_list
def get_seg_7_list(set_list=[], digits_list=seg_7_digits)-> list[set[str]]:
	if len(set_list) == 0:
		for i in digits_list:
			st = []
			n = i >> 1
			for j, c in enumerate(reversed('abcdefg')):
				bp = 1 << j
				st.insert(0, c if n & bp else '')
			set_list.append(tuple(st))
	return set_list
if __name__ == '__main__':
	print("SEG7_DICT={")
	for c in 'ABCDEFG':
		print(f"\t'{c.lower()}': Seg7.{c},")
	print("}")


	seg_7_set = get_seg_7_list()
	print("seg_7_array = (")
	for i, seg in enumerate(seg_7_set):
		print(f"\t{seg}, # {i:X}")
	print(')')