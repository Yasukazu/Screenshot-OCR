from seg_7_digits import Bit8, SEG7_ARRAY, BIT_TO_SEG_ELEM, BIT8_ARRAY

class Bin2(int):
	'''LSB is dot'''
	@property
	def dot_int(self)-> tuple[bool, int]:
		return bool(self & 1), self >> 1
	
	@property
	def of_int(self)-> int:
		return self >> 1
		
	@property
	def dot(self)-> bool:
		return bool(self & 1)
	
	@property
	def of_int(self)-> int:
		return self >> 1
		
	def to_bit8(self)-> Bit8 | None:
		of_int = self.of_int
		if not of_int:
			return Bit8.H if self.dot else None
		b8s = [b for b in BIT8_ARRAY if b.value & of_int]
		r = b8s[0]
		for i in range(1, len(b8s)):
			r |= b8s[i]
		return r | Bit8.H

if __name__ == '__main__':
	b2 = Bin2(1)
	print(b2)

	'''def __init__(self, n: int, dot=True):
		if not (0 <= n < len(SEG7_ARRAY)):
			raise ValueError("Out of hexadecimal range!")
		self.n = n << 1 | bool(dot)'''