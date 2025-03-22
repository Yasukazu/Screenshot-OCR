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

	def to_bit8(self)-> 'Seg7Bit8':
		from seg7bit8 import Seg7Bit8, SEG7BIT8_ARRAY
		of_int = self.of_int
		if not of_int:
			return Seg7Bit8.H if self.dot else None
		b8s = [b for b in SEG7BIT8_ARRAY if b.value & self.of_int]
		r = Seg7Bit8.NUL
		for i in b8s:
			r |= i
		return r


if __name__ == '__main__':
	b2 = Bin2(1)
	print(b2)
