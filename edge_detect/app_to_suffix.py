""" AppToSuffix StrEnum from env. val. """
APP_TO_SUFFIX_STR = "APP_TO_SUFFIX"
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, Type, Literal, get_args, TYPE_CHECKING, Union
from enum import Enum, StrEnum
import sys
from dotenv import find_dotenv, load_dotenv, dotenv_values
from os import environ as os_environ

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
from set_logger import set_logger
logger = set_logger(__name__)
from dot_env import DOTENV_INFO
if not DOTENV_INFO.is_valid:
	logger.error("Failed to load '.env' file.")
	raise ValueError("Failed to load '.env' file.")

ENV_PREFIX_STR = "IMAGE_FILTER"

def make_app_name_enum(app_to_suffix: str, env_prefix=ENV_PREFIX_STR, name="APP_NAME", module=__name__) -> Enum:
	"""Create APP_NAME Enum from a comma-separated string of 'name:integer' pairs.(like APP_NAME=A:1,B:2)"""
	if not app_to_suffix:
		env_var = f"{env_prefix}_{name}"
		app_to_suffix = os_environ.get(env_var)
		if not app_to_suffix:
			raise ValueError(f"'{env_var}' env. var. is missing!")
	mappings = {}
	for n, pair in enumerate(app_to_suffix.split(",")):
		try:
			k, v = pair.split(":")
			if not v or not k:
				raise ValueError(f"Invalid pair format [key:value]: no value or key: {pair}")
		except ValueError:
			raise ValueError("Invalid pair format [key:value] no delimiter: (:)")
		mappings[k.strip().upper()] = n + 1
	if not mappings:
		raise ValueError(f"'{app_to_suffix}' env. var. is empty!")
	return Enum(name, mappings, module)

def make_app_to_suffix_strenum(name_value_pair: str, name="APP_TO_SUFFIX", module=__name__) -> StrEnum:# tuple[str, dict[str, str]]:
	"""make StrEnum APP_TO_SUFFIX from a comma-separated string of 'key:value' pairs like: 'APP1:suffix1,APP2:suffix2'."""
	if not name:
		raise ValueError("'strenum_name' is empty!")
	if not name_value_pair:
		raise ValueError("'env_str' is empty!")

	mappings = {}
	for pair in name_value_pair.split(","):
		try:
			k, v = pair.split(":")
			if not v or not k:
				raise ValueError(f"Invalid pair format [key:value]: no value or key: {pair}")
		except ValueError:
			raise ValueError("Invalid pair format [key:value] no delimiter: (:)")
		name = k.strip().upper()
		mappings[name] = v.strip().lower()#.split('.')
	if not mappings:
		raise ValueError("mappings for name and value is empty")
	return StrEnum(name, mappings, module=module)

class AppToSuffix:
	""" partial emuration of StrEnum """
	def __init__(self, app_to_suffix: str, name="APP_TO_SUFFIX", module=__name__):
		self.app_to_suffix = make_app_to_suffix_strenum(app_to_suffix, name=name, module=module)
	def __getitem__(self, key: str)-> StrEnum:
		""" Getter: [] access like StrEnum """
		return getattr(self.app_to_suffix, key.upper())
	def items(self)-> dict[str, str]:
		""" Return items like StrEnum """
		result = {}
		for attr_name in dir(self.app_to_suffix):
			if not attr_name.startswith('_'):
				attr = getattr(self.app_to_suffix, attr_name)
				if hasattr(attr, 'value'):
					result[attr_name] = attr.value
		return result
class AppName:
	""" partial emuration of Enum """
	def __init__(self, app_to_suffix: str, name="APP_NAME", module=__name__):
		self.app_name = make_app_name_enum(app_to_suffix, name=name, module=module)
	def __getitem__(self, key: str)-> Enum:
		""" Getter: [] access like Enum """
		return getattr(self.app_name, key.upper())
	def items(self)-> dict[str, str]:
		""" Return items like Enum """
		result = {}
		for attr_name in dir(self.app_name):
			if not attr_name.startswith('_'):
				attr = getattr(self.app_name, attr_name)
				if hasattr(attr, 'value'):
					result[attr_name] = attr.value
		return result

class APP_TO_SUFFIX(StrEnum):
	TAIMEE = "jp.co.taimee"
	MERCARI = "jp.mercari.work"