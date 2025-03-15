from typing import Sequence

def conv_num_to_bin(num: int, fmt="%d")-> bytearray:
	n_str = fmt % num
	bb = bytearray(len(n_str))
	i = 0
	for i, c in enumerate(n_str):
		if c == '-':
			bb[i] = 16
		else:
			bb[i] = int(c, 16)
	return bb

class FormatNum:
	FORMAT = "%d"

	def __init__(self, num: int):
		self.num = num

	def conv_to_bin(self):
		return conv_num_to_bin(self.num, fmt=self.FORMAT)

class HexFormatNum(FormatNum):
	FORMAT = "%x"
	INDEX = '0123456789abcdef-'
	def conv_to_bin(self):
		n_str = self.FORMAT % self.num
		bb = bytearray(len(n_str))
		for i, c in enumerate(n_str):
			bb[i] = self.INDEX.index(c)
		return bb

def formatnums_to_bytearray(nn: Sequence[FormatNum | int])-> bytearray:
	bb = bytearray() #len(nn))
	for c in nn:
		match c:
			case FormatNum():
				bb += c.conv_to_bin()
			case int():
				bb += c.to_bytes()
			case _:
				raise TypeError("Needs to be int or FormatNum!")
	return bb



if __name__ == '__main__':
	from typing import Sequence
	from pprint import pp
	from enum import Enum

	class HexNum(Enum):
		A  = 10
		B  = 11
		C  = 12
		D  = 13
		E  = 14
		F  = 15
	def get_number_image(width: int, *nn: FormatNum): #, slant=0.25, padding=0.2):
		b_str = []
		for n in nn:
			b_s = n.conv_to_bin()
			b_str.extend(b_s)
		return b_str
	bstr = get_number_image(160, FormatNum(24),FormatNum(-1))
	pp(bstr)