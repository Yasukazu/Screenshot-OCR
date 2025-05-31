import logging
import os
from datetime import datetime, date
from pathlib import Path
logger = logging.getLogger(__name__)

SCREEN_BASE_DIR = 'SCREEN_BASE_DIR'
DEFAULT_SCREEN_BASE_DIR = 'screen-data' # default value
SCREEN_YEAR = 'SCREEN_YEAR'
SCREEN_MONTH = 'SCREEN_MONTH'

from dotenv import dotenv_values
# load_dotenv('.env', verbose=True, override=True) #, dotenv_path=Path.home() / '.env')
config = dotenv_values(".env")
if SCREEN_BASE_DIR not in config:
	screen_base_dir_config = os.environ.get(SCREEN_BASE_DIR)
	if screen_base_dir_config:
		config[SCREEN_BASE_DIR] = screen_base_dir_config
		logger.info("screen_base_dir_name is supplemented from environ value: %s", screen_base_dir_config)
	else:
		logger.warning("Environment variable '%s' is not set, using default value: %s", SCREEN_BASE_DIR, DEFAULT_SCREEN_BASE_DIR)
		config[SCREEN_BASE_DIR] = DEFAULT_SCREEN_BASE_DIR

def set_input_dir_root(root: Path | str):
	"""Set the root input directory path."""
	if isinstance(root, str):
		root = Path(root)
	if not root.is_absolute():
		root = Path().home() / root
	config[SCREEN_BASE_DIR] = str(root)
	logger.info("'%s' is set as: '%s'", SCREEN_BASE_DIR, config[SCREEN_BASE_DIR])
	return root

def get_input_dir_root():
	"""Get the root input directory path."""
	return Path().home() / config[SCREEN_BASE_DIR]

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

INPUT_EXT = '.png'

def path_pair_feeder(from_=1, to=31, input_ext='.png', output_ext='.tact', input_dir_root=get_input_dir_root()): #rng=range(0, 31)):
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

def get_last_month_path(direc: Path=get_input_dir_root(), year=0, month=0)-> Path:
	last_month = get_last_month()
	if not year:
		year = last_month.year
	if not month:
		month = last_month.month
	return direc / ("%d" % year) / ("%02d" % month)
	# MONTH_FORMAT.format(month)
	# ymstr = f"{year}{month:02}"

def get_year_month(year=0, month=0, config=config)-> date: # tuple[int, int]:
	year = year or int(config[SCREEN_YEAR])
	# if isinstance(SCREEN_MONTH, int) and SCREEN_MONTH > 0:
	month = month or int(config[SCREEN_MONTH])
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

def get_input_dir(year=0, month=0, input_dir_root=get_input_dir_root())-> Path:
	date = get_year_month(year=year, month=month)
	return input_dir_root / str(date.year) / ("%02d" % date.month)

def get_input_path(year=0, month=0, input_dir_root=get_input_dir_root())-> Path:
	ymstr = get_ymstr(year=year, month=month)
	return input_dir_root / ymstr

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