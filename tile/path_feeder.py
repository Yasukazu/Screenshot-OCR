# -- coding: utf-8 --
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

from input_dir import get_input_dir_root, get_year_month, get_last_month
from input_dir import FileExt
from calendar import monthrange



from typing import Generator, Iterator, Sequence
from input_dir import config, get_ymstr
class PathFeeder:
	def __init__(self, year=0, month=0, days: Sequence[int] | range | int=-1, input_type:FileExt=FileExt.PNG, input_dir=get_input_dir_root(), type_dir=True, config=config):
		last_date = get_year_month(year=year, month=month, config=config)
		self.year = last_date.year
		self.month = last_date.month
		input_path = (input_dir / str(self.year) / ("%02d" % self.month) / input_type.value.dir) if type_dir else input_dir / str(self.year) / ("%02d" % self.month) # / "%02d%02d" % (self.month, sel
		if not input_path.exists():
			raise ValueError(f"No path: {input_path}")
		self.input_path = input_path
		match days:
			case int():
				last_day = monthrange(self.year, self.month)[1]
				if days < 0:
					self.days = range(0, last_day)
				elif days < last_day:
					self.days = [days]
				else:
					raise ValueError(f'Last {days=} is over {last_day=}!')
			case _:
				self.days = days
		self.input_type = input_type
		self.type_dir = type_dir
	
	@property
	def dir(self)-> Path:
		return self.input_path

	@property
	def ext(self)-> str:
		return self.input_type.value.ext

	def glob(self, pattern='*'):
		for file in self.input_path.glob(pattern=pattern):
			yield file

	def    stem_gen(self, day: int, delim=''):
		return f"{day:02}" if self.type_dir else f"{self.month:02}{delim}{day:02}"

	def iter_stem(self, ext='.png'):
		for file in self.dir.iterdir():
			if file.is_file() and file.suffix == ext:
				yield file.stem
	def feed(self, padding=True, delim='') -> Iterator[tuple[int, str]]:
		for dy in range(monthrange(self.year, self.month)[1]):
			if dy in self.days:
				stem = self.stem_gen(dy, delim=delim)
				input_fullpath = self.input_path / (stem + self.input_type.value.ext) # ''.join([s for s in stem if s!= delim])
				if not padding and not input_fullpath.exists():
					raise FileNotFoundError(f"{input_fullpath=} does not exist!")
				yield dy, stem
			else:
				yield dy, '' #stem if input_fullpath.exists() else ''
		'''else:
			stems = [f.stem for f in self.input_path.glob("??" + self.input_type.value.ext)]
			yield from sorted(stems)'''

	@property
	def first_name(self): #-> str:
		for n, stem in self.feed(padding=False):
			return stem
		return ''

	@property
	def first_fullpath(self)-> Path | None:
		stem = None
		for day, stem in self.feed(padding=False):
			break
		return self.dir / (stem + self.ext) if stem else None

from contextlib import closing
import txt_lines_db
class DbPathFeeder(PathFeeder):
	from app_type import AppType
	img_file_ext = '.png'

	def __init__(self, year=0, month=0, days=-1, input_type = FileExt.PNG, input_dir=get_input_dir_root(), type_dir=False, app_type=AppType.T, config=config, db_fullpath=txt_lines_db.sqlite_fullpath()):
		super().__init__(year, month, days, input_type, input_dir, type_dir, config)
		self.app_type = app_type
		self.conn = txt_lines_db.connect(db_fullpath=db_fullpath)
		
	@property
	def table_name(self):#, month: int):
		return txt_lines_db.get_table_name(self.month)

	def table_exists(self, month: int=0,
		sql = """SELECT EXISTS (
			SELECT name
			FROM sqlite_schema 
			WHERE type='table' AND name = '?')"""):
		if not month:
			month = self.month
		with closing(self.conn.cursor()) as cur:
			one = cur.execute(sql, self.table_name).fetchone()
			return bool(one)
	
	def feed(self, padding=False, delim='', day_to_stem={}) -> Iterator[tuple[int, str]]:
			tbl_name = txt_lines_db.get_table_name(self.month)
			day_list = f"({','.join([str(d + 1) for d in self.days])})"
			sql = f"SELECT `day`, `stem` FROM `{tbl_name}`" + (f"WHERE day IN {day_list} AND app = {str(self.app_type.value)}") + " ORDER BY `day`;"
			with closing(self.conn.cursor()) as cur:
				for r in cur.execute(sql):
					day_to_stem[r[0]] = r[1]
				# day_to_stem = {r[0]: r[1] for r in rr}
			if padding:
				for dy in range(1, monthrange(self.year, self.month)[1] + 1):
					yield dy, day_to_stem[dy] if dy in day_to_stem else ''
			else:
				for dy in day_to_stem:
					yield dy, day_to_stem[dy]

def path_feeder(year=0, month=0, from_=1, to=-1, input_type:FileExt=FileExt.PNG, padding=True, input_dir_root=get_input_dir_root())-> Generator[tuple[Path | None, str, int], None, None]:
	'''to=0:glob, -1:end of month
	returns: directory, filename, day'''
	last_date = get_last_month(year=year, month=month) # f"{year}{month:02}"
	year = last_date.year
	month = last_date.month
	input_path = input_dir_root / str(year) / ("%02d" % month) / input_type.value.dir
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



from enum import StrEnum

TIFF_EXT = '.tif'
def get_imgnum_sfx(n):
	return '-img%02d' % n

def get_tiff_fullpath(year=0, month=0, input_dir=get_input_dir_root())-> Path:
	ym_str = get_ymstr(year=year, month=month, sep=True)
	return input_dir / ''.join([ym_str, get_imgnum_sfx(32), TIFF_EXT])

if __name__ == '__main__':
	from pprint import pp
	feeder = DbPathFeeder()
	files = [f for f in feeder.feed()]
	pp(files)
