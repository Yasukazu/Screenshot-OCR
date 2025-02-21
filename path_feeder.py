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

YEAR_FORMAT = "{:04}"
MONTH_FORMAT = "{:02}"

def get_last_month_path(year=0, month=0)-> Path:
	last_month = get_last_month()
	if not year:
		year = last_month.year
	if not month:
		month = last_month.month
	return input_dir / ("%d" % year) / ("%02d" % month)
	# MONTH_FORMAT.format(month)
	# ymstr = f"{year}{month:02}"
from datetime import date
def get_year_month(year=0, month=0)-> date: # tuple[int, int]:
	last_month = get_last_month()
	if not year:
		year = last_month.year
	if not month:
		month = last_month.month
	return date(year, month, 1)

def get_ymstr(year=0, month=0, sep=False)-> str:
	last_month = get_last_month()
	if not year:
		year = last_month.year
	if not month:
		month = last_month.month
	sepr = '-' if sep else ''
	return f"{year}{sepr}{month:02}"

def get_input_path(year=0, month=0)-> Path:
	ymstr = get_ymstr(year=year, month=month)
	return input_dir / ymstr

from typing import Generator, Iterator
class PathFeeder:
	def __init__(self, year=0, month=0, from_=1, to=-1, input_type:FileExt=FileExt.PNG):
		last_date = get_year_month(year=year, month=month)
		year = last_date.year
		month = last_date.month
		input_path = input_dir / str(year) / ("%02d" % month) / input_type.value.dir
		if not input_path.exists():
			raise ValueError(f"No path: {input_path}")
		self.input_path = input_path
		self.to = monthrange(year, month)[1] if to < 0 else to
		self.from_ = from_
		self.input_type = input_type
	
	@property
	def dir(self)-> Path:
		return self.input_path

	@property
	def ext(self)-> str:
		return self.input_type.value.ext

	def feed(self, padding=True) -> Iterator[str]:
		if (self.to + 1) - self.from_ > 0:
			for day in range(self.from_, self.to + 1):
				stem = f"{day:02}"
				input_fullpath = self.input_path / (stem + self.input_type.value.ext)
				if not padding:
					if input_fullpath.exists():
						yield stem
				else:
					yield stem if input_fullpath.exists() else ''
		else:
			stems = [f.stem for f in self.input_path.glob("??" + self.input_type.value.ext)]
			yield from sorted(stems)

	@property
	def first_fullpath(self)-> Path | None:
		stem = None
		for stem in self.feed(padding=False):
			break
		return self.dir / (stem + self.ext) if stem else None

def path_feeder(year=0, month=0, from_=1, to=-1, input_type:FileExt=FileExt.PNG, padding=True)-> Generator[tuple[Path | None, str, int], None, None]:
	'''to=0:glob, -1:end of month
	returns: directory, filename, day'''
	last_date = get_year_month(year=year, month=month) # f"{year}{month:02}"
	year = last_date.year
	month = last_date.month
	input_path = input_dir / int(year) / ("%02d" % month) / input_type.value.dir
	#if direc: input_path = input_path / direc
	if to < 0:
		to = monthrange(year, month)[1]
	if (to + 1) - from_ > 0:
		for day in range(from_, to + 1):
			stem = f"{day:02}" #'f'2025-01-{day:02}'
			input_fullpath = input_path / (stem + input_type.value.ext)
			if input_fullpath.exists():
				yield input_path, stem, day
			else:
				if padding:
					yield None, stem, day
	else: # files = list(input_path.glob("*"))#.sort()
		stems = [f.stem for f in input_path.glob("??" + input_type.value.ext)]
		for stem in sorted(stems):
			yield input_path, stem, 0

import datetime
from datetime import date

def get_last_month(year=0, month=0)-> date:
	if not (year and month):
		today = datetime.date.today()
		first = today.replace(day=1)
		last_month = first - datetime.timedelta(days=1)
		ym = last_month.strftime("%Y,%m").split(',')
		y, m = (int(i) for i in ym)
		year = year or y
		month = month or m
	return date(year=year, month=month, day=1) # YearMonth(*iym)

def get_cur_month()-> date: #YearMonth:
	today = datetime.date.today()
	ym = today.strftime("%Y,%m").split(',')
	y, m = (int(i) for i in ym)
	return date(y, m, 1) # YearMonth(*iym)

from enum import StrEnum

TIFF_EXT = '.tif'
def get_imgnum_sfx(n):
	return '-img%02d' % n

def get_tiff_fullpath(year=0, month=0)-> Path:
	ym_str = get_ymstr(year=year, month=month, sep=True)
	return input_dir / ''.join([ym_str, get_imgnum_sfx(32), TIFF_EXT])

if __name__ == '__main__':
	path_feeder = PathFeeder()
	first_fulpath = path_feeder.first_fullpath
	print(first_fullpath)
