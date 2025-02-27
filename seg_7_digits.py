seg_7_digits = (
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
seg_7_array = (
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
def get_seg_7_list(set_list=[], digits_list=seg_7_digits)-> list[set[str]]:
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
				if n & bp:
					st.insert(0, c)
			set_list.append(tuple(st))
	return set_list
if __name__ == '__main__':
	seg_7_set = get_seg_7_list()
	print("seg_7_array = (")
	for i, seg in enumerate(seg_7_set):
		print(f"\t{seg}, # {i:X}")
	print(')')