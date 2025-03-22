from enum import Flag

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
