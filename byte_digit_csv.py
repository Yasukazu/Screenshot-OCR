# %%
# 
'''digits = (
  0b11111100, # 0
  0b01100000, # 1
  0b11011010, # 2
  0b11110010, # 3
  0b01100110, # 4
  0b10110110, # 5
  0b10111110, # 6
  0b11100000, # 7
  0b11111110, # 8
  0b11110110, # 9
)'''
import csv
from enum import Enum
from pprint import pp
from pathlib import Path
from path_feeder import path_feeder
import os
CWD = os.getcwd()
SEVEN_SEG_STEM = '7-seg'
SEVEN_SEG_MAX = 16
# SEVEN_SEG_SIZE = 17
CSV_EXT = '.csv'
PKL_EXT = '.pkl'
class Ext(Enum):
	CSV = '.csv'
	PKL = '.pkl'

# type str_int_dict = dict[str, int]

def get_fullpath(stem: str, ext: Ext, parent: Path=Path(CWD)):
	return parent / (stem + ext.value)
SEVEN_SEG_NUM_STEM = '7-seg-num'

def load_7_seg_num_csv_as_dict(lst=[])-> list[dict[str, int]]:
	if len(lst) == 0:
		fullpath = get_fullpath(stem=SEVEN_SEG_NUM_STEM, ext=Ext.CSV)
		with fullpath.open() as csv_file:
			reader = csv.DictReader(csv_file)
			for row in reader:
				lst.append(row)
	return lst

def enbyte(row: dict[str, int])-> int:
	ac = 0
	for c in 'abcdefg':
		if row[c] == '1':
			ac |= 1
		ac <<= 1
	return ac

def get_digit_list(
		seven_seg_digits = load_7_seg_num_csv_as_dict()
	)-> list[dict[str, int]]:
	digits_list = []
	for row in seven_seg_digits:
		bt = enbyte(row)
		digits_list.append(bt)
	return digits_list


if __name__ == '__main__':
	digits_list = get_digit_list()
	print("digits = (")
	print("\t# abcdefgh")
	for n, i in enumerate(digits_list):
		print(f"\t0b{i:08b}, # {n:02X}:{i}")
	print(')')