import os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import logbook
logbook.StreamHandler(sys.stdout,
	format_string='{record.time:%Y-%m-%d %H:%M:%S.%f} {record.level_name} {record.filename}:{record.lineno}: {record.message}').push_application()
logger = logbook.Logger(__file__)
logger.level = logbook.INFO

def get_screen_base_dir():
	screen_base_dir_name = os.getenv('SCREEN_BASE_DIR')
	return Path(screen_base_dir_name) if screen_base_dir_name else Path(__file__).parent

SCREEN_BASE_DIR = 'SCREEN_BASE_DIR'
SCREEN_YEAR = 'SCREEN_YEAR'
SCREEN_MONTH = 'SCREEN_MONTH'

from dotenv import dotenv_values

config_dict = dotenv_values(".env")
logger.info("config from dotenv_values: {}", config_dict)
# is_dotenv_loaded = load_dotenv('.env', verbose=True)

for k in [SCREEN_BASE_DIR, SCREEN_YEAR, SCREEN_MONTH]:
	config_dict[k] = config_dict.get(k) or os.environ.get(k)

try:
	input_dir_root = Path(config_dict[SCREEN_BASE_DIR])
except Exception as exc:
	raise ValueError(f"{SCREEN_BASE_DIR} is not set.") from exc

if not input_dir_root.exists():
	raise ValueError(f"`{input_dir_root=}` for {SCREEN_BASE_DIR} does not exist!")

def check_y_m(key):
	value = config_dict.get(key)
	if not value:
		config_dict[key] = '0'
		return
	if not value.isdigit():
		logger.error("Value: %s (for %s) must be digit!", value, key)
		raise ValueError(f"Invalid {value=} for {key=}!")

for key in [SCREEN_YEAR, SCREEN_MONTH]:
	check_y_m(key)

input_ext = '.png'
