# -- coding: utf-8 --
import os, sys
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(stdout_handler)
format_output = logging.Formatter('%(levelname)s : %(name)s : %(message)s : %(asctime)s') # <-
stdout_handler.setFormatter(format_output)
home_dir = os.path.expanduser('~')
home_path = Path(home_dir)

SCREEN_BASE_DIR = 'SCREEN_BASE_DIR'
SCREEN_YEAR = 'SCREEN_YEAR'
SCREEN_MONTH = 'SCREEN_MONTH'

from dotenv import dotenv_values, load_dotenv
# load_dotenv('.env', verbose=True, override=True) #, dotenv_path=Path.home() / '.env')
config = dotenv_values(".env")

logger.info("config from dotenv_values: %s", config)
# is_dotenv_loaded = load_dotenv('.env', verbose=True)

screen_base_dir = config.get(SCREEN_BASE_DIR) or os.environ.get(SCREEN_BASE_DIR)
if not screen_base_dir:
	config[SCREEN_BASE_DIR] = screen_base_dir = str(home_path)
	logger.warning("'screen_base_dir' is set as '%s' since environment variable '%s' is not set.", screen_base_dir, SCREEN_BASE_DIR)
input_dir_root = Path(screen_base_dir)
if not input_dir_root.exists():
	raise ValueError(f"`{input_dir_root=}` for {SCREEN_BASE_DIR} does not exist!")

def check_y_m(key: str):
	value = config.get(key) or os.environ.get(key) or '0'
	if not value.isdigit():
		logger.error("Value: %s (for %s) must be digit!", value, key)
		raise ValueError(f"Invalid {value=} for {key=}!")
	else:
		value = int(value)
		if value < 0:
			logger.error("Value: %s (for %s) must be larger than or equal to 0!", value, key)
			raise ValueError(f"Invalid {value=} for {key=}")
		config[key] = value

for key in [SCREEN_YEAR, SCREEN_MONTH]:
	check_y_m(key)

input_ext = '.png'

def path_pair_feeder(from_=1, to=31, input_ext='.png', output_ext='.tact'): #rng=range(0, 31)):
	for day in range(from_, to + 1):
		input_filename = f'2025-01-{day:02}{input_ext}'
		input_fullpath = input_dir_root / input_filename
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
	TIFF = ExtDir('.tif', 'tiff')
	BTM_TXT = ExtDir('.btm.txt', 'png')
	TOP_TXT = ExtDir('.top.txt', 'png')

def ext_to_dir(ext: str)-> str:
	assert ext[0] == '.'
	return ext[1:]

from collections import namedtuple
from calendar import monthrange

YearMonth = namedtuple('YearMonth', ['year', 'month'] )

YEAR_FORMAT = "{:04}"
MONTH_FORMAT = "{:02}"

def get_last_month_path(dir: Path=input_dir_root, year=0, month=0)-> Path:
	last_month = get_last_month()
	if not year:
		year = last_month.year
	if not month:
		month = last_month.month
	return dir / ("%d" % year) / ("%02d" % month)
	# MONTH_FORMAT.format(month)
	# ymstr = f"{year}{month:02}"
from datetime import date
def get_year_month(year=0, month=0, config=config)-> date: # tuple[int, int]:
	year = int(config[SCREEN_YEAR])
	# if isinstance(SCREEN_MONTH, int) and SCREEN_MONTH > 0:
	month = int(config[SCREEN_MONTH])
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
	return input_dir_root / ymstr

from typing import Generator, Iterator, Sequence
class PathFeeder:
	def __init__(self, year=0, month=0, days: Sequence[int] | range | int=-1, input_type:FileExt=FileExt.PNG, input_dir=input_dir_root, type_dir=True, config=config):
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

	def __init__(self, year=0, month=0, days=-1, input_type = FileExt.PNG, input_dir=input_dir_root, type_dir=False, app_type=AppType.T, config=config, db_fullpath=txt_lines_db.sqlite_fullpath()):
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

def path_feeder(year=0, month=0, from_=1, to=-1, input_type:FileExt=FileExt.PNG, padding=True)-> Generator[tuple[Path | None, str, int], None, None]:
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
	return input_dir_root / ''.join([ym_str, get_imgnum_sfx(32), TIFF_EXT])

if __name__ == '__main__':
	from logger import set_logger
	set_logger()
	from pprint import pp
	feeder = DbPathFeeder()
	files = [f for f in feeder.feed()]
	pp(files)
