seg_8_digits = (
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
def get_seg_8_set(set_list=[]):
	if len(set_list) == 0:
		for i in seg_8_digits:
			st = set()
			n = i >> 1
			for j, c in enumerate(reversed('abcdefg')):
				bp = 1 << j
				if n & bp:
					st.add(c)
			set_list.append(st)
	return set_list