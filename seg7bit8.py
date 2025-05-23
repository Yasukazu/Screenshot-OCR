from enum import Flag
from types import MappingProxyType

class Seg7Bit8(Flag):
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


SEG7BIT8_ARRAY = (
	Seg7Bit8.A,
	Seg7Bit8.B,
	Seg7Bit8.C,
	Seg7Bit8.D,
	Seg7Bit8.E,
	Seg7Bit8.F,
	Seg7Bit8.G,
	Seg7Bit8.H
)

SEG7BIT8_VALUE_ARRAY = (
	Seg7Bit8.A.value,
	Seg7Bit8.B.value,
	Seg7Bit8.C.value,
	Seg7Bit8.D.value,
	Seg7Bit8.E.value,
	Seg7Bit8.F.value,
	Seg7Bit8.G.value,
	Seg7Bit8.H.value,
)

from bin2 import Bin2
def bin2_to_seg7bit8(b2: Bin2)-> Seg7Bit8:
	of_int = b2.of_int
	if not of_int:
		return Seg7Bit8.H if b2.dot else Seg7Bit8.NUL
	b8s = [b for b in SEG7BIT8_ARRAY if b.value & of_int]
	r = Seg7Bit8.NUL
	for i in b8s:
		r |= i
	return r

BIN1_TO_SEG7BIT8 = MappingProxyType({
	Seg7Bit8.A.value: Seg7Bit8.A,
	Seg7Bit8.B.value: Seg7Bit8.B,
	Seg7Bit8.C.value: Seg7Bit8.C,
	Seg7Bit8.D.value: Seg7Bit8.D,
	Seg7Bit8.E.value: Seg7Bit8.E,
	Seg7Bit8.F.value: Seg7Bit8.F,
	Seg7Bit8.G.value: Seg7Bit8.G,
	Seg7Bit8.H.value: Seg7Bit8.H,
})
