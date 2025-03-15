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

def conv_to_bin(num: float, fmt="%d")-> bytearray:
	INDEX = '0123456789abcdef-.'
	n_str = fmt % num
	bb = bytearray(len(n_str))
	for i, c in enumerate(n_str):
		bb[i] = INDEX.index(c)
	return bb

class FormatNum:
	FORMAT = "%d"

	def __init__(self, num: float, fmt=None):
		self.num = num
		if fmt:
			self.FORMAT = fmt

	def conv_to_bin(self):
		return conv_to_bin(self.num, fmt=self.FORMAT)

class HexFormatNum(FormatNum):
	FORMAT = "%x"
	def conv_to_bin(self):
		return conv_to_bin(self.num, fmt=self.FORMAT)
class FloatFormatNum(FormatNum):
	FORMAT = "%f"
	def conv_to_bin(self):
		return conv_to_bin(self.num, fmt=self.FORMAT)

def formatnums_to_bytearray(nn: Sequence[FormatNum | int], fmt=None)-> bytearray:
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
	from pprint import pp
	ff = [FloatFormatNum(0.1, "%.1f")]
	bb = formatnums_to_bytearray(ff)
	pp(bb)

	from typing import Sequence
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