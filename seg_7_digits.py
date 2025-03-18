from typing import Sequence, Callable
from types import MappingProxyType
from enum import Flag, IntEnum
from strok7 import SpPair, SegElem

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

class Bit8(Flag):
	'''MSB is A, LSB is for comma'''
	A = 1 << 7
	B = 1 << 6
	C = 1 << 5
	D = 1 << 4
	E = 1 << 3
	F = 1 << 2
	G = 1 << 1
	H = 1 # comma / period /dot
	NUL = 0

BIT8_ARRAY = (
	Bit8.A,
	Bit8.B,
	Bit8.C,
	Bit8.D,
	Bit8.E,
	Bit8.F,
	Bit8.G,
	Bit8.H
)

SEG7_TO_SP_PAIR = MappingProxyType({
	Bit8.A: SpPair.A,
	Bit8.B: SpPair.B,
	Bit8.C: SpPair.C,
	Bit8.D: SpPair.D,
	Bit8.E: SpPair.E,
	Bit8.F: SpPair.F,
	Bit8.G: SpPair.G,
	Bit8.H: SpPair.H, # dot
})

BIT_TO_SP_PAIR = {
	Bit8.A.value: SpPair.A,
	Bit8.B.value: SpPair.B,
	Bit8.C.value: SpPair.C,
	Bit8.D.value: SpPair.D,
	Bit8.E.value: SpPair.E,
	Bit8.F.value: SpPair.F,
	Bit8.G.value: SpPair.G,
	Bit8.H.value: SpPair.H,
}

BIT_TO_SEG_ELEM = {
	Bit8.A.value: SegElem.A,
	Bit8.B.value: SegElem.B,
	Bit8.C.value: SegElem.C,
	Bit8.D.value: SegElem.D,
	Bit8.E.value: SegElem.E,
	Bit8.F.value: SegElem.F,
	Bit8.G.value: SegElem.G,
	Bit8.H.value: SegElem.H,
}

BIT8_TO_SEG_ELEM = {
	Bit8.A: SegElem.A,
	Bit8.B: SegElem.B,
	Bit8.C: SegElem.C,
	Bit8.D: SegElem.D,
	Bit8.E: SegElem.E,
	Bit8.F: SegElem.F,
	Bit8.G: SegElem.G,
	Bit8.H: SegElem.H,
}

def str_to_seg_elems(n_s: str)-> list[Sequence[SegElem]]:
	INDEX = '0123456789abcdef-'
	n_str = n_s + '\0'
	bb = []
	i = 0
	while i < len(n_str) - 1:
		b = INDEX.index(n_str[i]) << 1
		if n_str[i + 1] == '.':
			b += 1
			i += 1
		b8 = bin2_to_bit8(b)
		bb += [expand_bit8_to_seg_elems(b8)]
		i += 1
	return bb

def expand_bin_to_seg_elems(bn: int)-> Sequence[SegElem]:
	return tuple(BIT_TO_SEG_ELEM[bit] for bit in (Bit8.A.value, Bit8.B.value, Bit8.C.value, Bit8.D.value, Bit8.E.value, Bit8.F.value, Bit8.G.value, Bit8.H.value) if bit & bn)

def expand_bit8_to_seg_elems(bits: Bit8)-> Sequence[SegElem]:
	# bits = bin2_to_bit8(bn)
	return tuple(BIT8_TO_SEG_ELEM[bit] for bit in (Bit8.A, Bit8.B, Bit8.C, Bit8.D, Bit8.E, Bit8.F, Bit8.G, Bit8.H) if bit & bits)

def expand_bin2_to_seg_elems(bn: int)-> Sequence[SegElem]:
	bits = bin2_to_bit8(bn)
	return tuple(BIT_TO_SEG_ELEM[bit] for bit in (Bit8.A, Bit8.B, Bit8.C, Bit8.D, Bit8.E, Bit8.F, Bit8.G, Bit8.H) if bit & bits)

def expand_bin_to_sp_pairs(bn: int)-> Sequence[SpPair]:
	return tuple(BIT_TO_SP_PAIR[bit] for bit in (Bit8.A.value, Bit8.B.value, Bit8.C.value, Bit8.D.value, Bit8.E.value, Bit8.F.value, Bit8.G.value, Bit8.H.value) if bit & bn)

def expand_bin_to_xy_list_list(bn: int)-> list[list[tuple[int, int]]]:
	return [SpPair.expand_to_xy_list(spp) for spp in expand_bin_to_sp_pairs(bn)]

def expand_to_sp_pairs(seg7: Bit8)-> Sequence[SpPair]:
	return tuple(SEG7_TO_SP_PAIR[seg] for seg in (Bit8.A, Bit8.B, Bit8.C, Bit8.D, Bit8.E, Bit8.F, Bit8.G) if seg7 & seg)

def expand_to_xy_list_list(seg7: Bit8)-> list[list[tuple[int, int]]]:
	return [SpPair.expand_to_xy_list(spp) for spp in expand_to_sp_pairs(seg7)]

def c_to_seg_7(c: str, C_TO_SEG7 = {
		'a': Bit8.A,
		'b': Bit8.B,
		'c': Bit8.C,
		'd': Bit8.D,
		'e': Bit8.E,
		'f': Bit8.F,
		'g': Bit8.G,
})-> Callable[[str], Bit8]:
	return C_TO_SEG7[c]

SEG7_ARRAY = (
	(Bit8.A | Bit8.B | Bit8.C | Bit8.D | Bit8.E | Bit8.F),
	(Bit8.B | Bit8.C),
	(Bit8.A | Bit8.B | Bit8.D | Bit8.E | Bit8.G),
	(Bit8.A | Bit8.B | Bit8.C | Bit8.D | Bit8.G),
	(Bit8.B | Bit8.C | Bit8.F | Bit8.G),
	(Bit8.A | Bit8.C | Bit8.D | Bit8.F | Bit8.G),
	(Bit8.A | Bit8.C | Bit8.D | Bit8.E | Bit8.F | Bit8.G),
	(Bit8.A | Bit8.B | Bit8.C),
	(Bit8.A | Bit8.B | Bit8.C | Bit8.D | Bit8.E | Bit8.F | Bit8.G),
	(Bit8.A | Bit8.B | Bit8.C | Bit8.F | Bit8.G),
	(Bit8.A | Bit8.B | Bit8.C | Bit8.E | Bit8.F | Bit8.G),
	(Bit8.C | Bit8.D | Bit8.E | Bit8.F | Bit8.G),
	(Bit8.D | Bit8.E | Bit8.G),
	(Bit8.B | Bit8.C | Bit8.D | Bit8.E | Bit8.G),
	(Bit8.A | Bit8.D | Bit8.E | Bit8.F | Bit8.G),
	(Bit8.A | Bit8.E | Bit8.F | Bit8.G),
	(Bit8.G),
	Bit8.H
)

def hex_to_bit8(n: int)-> Bit8:
	'''16 for hyphen/minus, 17 for comma/period'''
	if not (0 <= n < len(SEG7_ARRAY)):
		raise ValueError("Out of hexadecimal range!")
	return SEG7_ARRAY[n]

def bin2_to_bit8(n: int)-> Bit8:
	'''LSB is for dot'''
	dot = n & 1
	n >>= 1
	if not (0 <= n < len(SEG7_ARRAY)):
		raise ValueError("Out of hexadecimal range!")
	bit8 = SEG7_ARRAY[n]
	if dot:
		bit8 |= Bit8.H
	return bit8

def bin_to_bit8(b: int)-> Bit8:
	if not 0 < b < 256:
		raise ValueError("Needs non-nul byte!")
	seg7_list = [seg for seg in (Bit8.A, Bit8.B, Bit8.C, Bit8.D, Bit8.E, Bit8.F, Bit8.G, Bit8.H) if b & seg.value]
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