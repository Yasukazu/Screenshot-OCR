""" APP_NAME Enum by class definition;AppName class to enulate a class member """
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, Type, Literal, get_args, TYPE_CHECKING
from enum import Enum, IntEnum
import sys
from dotenv import find_dotenv, load_dotenv, dotenv_values
from os import environ as os_environ
from enum import StrEnum

class APP_SYM(IntEnum):
	""" Application Symbol"""
	TM = 1
	MC = 2

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
from set_logger import set_logger
import sys
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
logger = set_logger(__name__)
ENV_FILENAME = ".env"
try:
	dotenv_path = find_dotenv(ENV_FILENAME, raise_error_if_not_found=True)
	logger.info("Loading .env file: %s", dotenv_path)
	with open(dotenv_path, 'r') as f:
		logger.info("Contents of .env file:\n%s", f.read())
		dotenv_values = dotenv_values(dotenv_path)
		logger.info("Loaded .env values: %s", dotenv_values)
		os_environ |= dotenv_values
	# load_dotenv(dotenv_path, override=True)
except Exception as e:
	logger.info("Failed to load .env file: %s", e)
MAIN_SETTINGS_FILENAME = os_environ.get("IMAGE_FILTER_MAIN_SETTINGS_FILENAME", "image-filter-main-settings.toml")
# Environment variables will be loaded after function definitions

# import typed_settings as tst
def image_area_param_names():
	from image_filter import ImageAreaParamName
	return list(ImageAreaParamName)
ENV_PREFIX = "IMAGE_FILTER"
def make_app_name_enum(prefix=ENV_PREFIX, enum_name="APP_NAME", module="__main__") -> Enum:
	"""Create APP_NAME Enum from a comma-separated string of 'name:integer' pairs.(like APP_NAME=A:1,B:2)"""
	env_var = f"{prefix}_{enum_name}"
	env_str = os_environ.get(env_var)
	if not env_str:
		raise ValueError(f"'{env_var}' env. var. is missing!")
	mappings = {}
	for pair in env_str.split(","):
		try:
			k, v = pair.split(":")
			if not v or not k:
				raise ValueError(f"Invalid pair format [key:value]: no value or key: {pair}")
		except ValueError:
			raise ValueError("Invalid pair format [key:value] no delimiter: (:)")
		mappings[k.strip().upper()] = int(v)
	if not mappings:
		raise ValueError(f"'{env_var}' env. var. is empty!")
	return Enum(enum_name, mappings, module)

APP_NAME = make_app_name_enum(module='__main__')
APP_NAMES: list[str] = [n.name for n in APP_NAME]
APP_NAME_LITERAL = Literal[*APP_NAMES]

from enum import StrEnum
def make_app_to_suffix_strenum(prefix=ENV_PREFIX, enum_name="APP_TO_SUFFIX", make_strenum=True, name_to_suffix: str = None) -> StrEnum|str:# tuple[str, dict[str, str]]:
	"""Create StrEnum APP_NAME from a comma-separated string of 'key:value' pairs."""
	env_var = f"{prefix}_{enum_name}"
	env_str = name_to_suffix or os_environ.get(env_var)
	if not env_str:
		raise ValueError(f"'{env_var}' env. var. is missing!")
	mappings = {}
	for pair in env_str.split(","):
		try:
			k, v = pair.split(":")
			if not v or not k:
				raise ValueError(f"Invalid pair format [key:value]: no value or key: {pair}")
		except ValueError:
			raise ValueError("Invalid pair format [key:value] no delimiter: (:)")
		name = k.strip().upper()
		if name not in APP_NAMES:
			raise ValueError(f"Invalid app name: {name}")
		mappings[name] = v.strip().lower()#.split('.')
	if not mappings:
		raise ValueError(f"'{env_var}' env. var. is empty!")
	return StrEnum(enum_name, mappings) if make_strenum else env_str #(enum_name, mappings)

class AppToSuffix:
	""" partial emuration of StrEnum """
	def __init__(self, app_to_suffix: str):
		self.app_to_suffix = make_app_to_suffix_strenum(name_to_suffix=app_to_suffix, make_strenum=True)
	def __getitem__(self, key: str):
		""" Getter: [] access like StrEnum """
		return self.app_to_suffix[key.upper()]
	def items(self)-> dict[str, str]:
		""" Return items like StrEnum """
		return {member.name: member.value for member in self.app_to_suffix}
