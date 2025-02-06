import os
from pathlib import Path

home_dir = os.path.expanduser('~')
home_path = Path(home_dir)
input_dir = home_path / 'Documents' / 'screen' # / '202501'
assert input_dir.exists()

def path_pair_feeder(from_=1, to=31, input_ext='.png', output_ext='.tact'): #rng=range(0, 31)):
	for day in range(from_, to + 1):
		input_filename = f'2025-01-{day:02}{input_ext}'
		input_fullpath = input_dir / input_filename
		if not input_fullpath.exists():
			continue
		input_path_noext, _ext = os.path.splitext(input_fullpath)
		output_path = Path(input_path_noext + output_ext)
		assert not output_path.exists()
		yield input_fullpath, output_path

from collections import namedtuple
ExtDir = namedtuple('ExtDir', ['ext', 'dir'])
from enum import Enum
class FileExt(Enum):
	TXT = ExtDir('.txt', 'txt')
	PNG = ExtDir('.png', 'png')
	QPNG = ExtDir('.png', 'qpng')

def ext_to_dir(ext: str)-> str:
	assert ext[0] == '.'
	return ext[1:]

from collections import namedtuple
from calendar import monthrange

YearMonth = namedtuple('YearMonth', ['year', 'month'] )

def path_feeder(year=0, month=0, from_=1, to=-1, input_type:FileExt=FileExt.PNG, pad=True): 
	'''to=0:glob, -1:end of month'''
	last_month = get_last_month()
	if not year:
		year = last_month.year
	if not month:
		month = last_month.month
	ymstr = f"{year}{month:02}"
	input_path = input_dir / ymstr / input_type.value.dir
	#if direc: input_path = input_path / direc
	if to < 0:
		to = monthrange(year, month)[1]
	if (to + 1) - from_ > 0:
		for day in range(from_, to + 1):
			node = f"{day:02}" #'f'2025-01-{day:02}'
			input_fullpath = input_path / (node + input_type.value.ext)
			if input_fullpath.exists():
				yield input_fullpath , node, day
			else:
				if pad:
					yield None , node, day
	else:
		files = list(input_path.glob("*"))#.sort()
		for file in sorted(files):
			yield file, file.stem, 0

import datetime

def get_last_month()-> YearMonth:
	'''returns (year, month)'''
	today = datetime.date.today()
	first = today.replace(day=1)
	last_month = first - datetime.timedelta(days=1)
	ym = last_month.strftime("%Y,%m").split(',')
	iym = [int(i) for i in ym]
	return YearMonth(*iym)