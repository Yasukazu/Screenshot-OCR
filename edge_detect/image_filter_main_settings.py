""" MainSettings by DataclassBinder"""
from pathlib import Path
import tomllib
from typing import Any, Callable, Iterator, Literal, Optional
from dataclasses import dataclass, field, fields
from enum import Enum
from dataclass_binder import Binder
from set_logger import set_logger
from functools import cached_property
from simple_parsing.helpers import Serializable
# from simple_parsing.helpers.serializable import yaml_serialization


logger = set_logger(__name__)
# import typed_settings as tst
def image_area_param_names():
	from image_filter import ImageAreaParamName
	return list(ImageAreaParamName)
def app_names():
	#from image_filter import APP_NAME
	return ['TAIMEE', 'MERCARI']#[n.name.lower() for n in APP_NAME]
def area_param_names():
	return ['HEADING', 'SHIFT', 'BREAKTIME', 'PAYSTUB', 'SALARY']
def default_factories():
	return [app_names, area_param_names]

case_mode = Literal['upper', 'lower']
# @yaml_serialization
from serde import serialize, deserialize

@serialize
@deserialize
@dataclass(kw_only=True)
class MainSettings(Serializable):
	"""
	Extract/OCR paystub text from an image file: Files for OCR by 'files' option may be specified with app-name-suffix in wildcard(glob pattern matching like '--files *.<APP_NAME>*.png') or by 'shot-month' option (like '--shot_month -1' for last month, 0 for current month, other positive value for month number: Jan. is 1, Dec. is 12, ...) and 'app' option (like '--app taimee')
	"""

	'''@classmethod
	def from_dict(cls, toml_dict: dict[str, Any]) -> 'MainSettings':
		return Binder(MainSettings).bind(toml_dict) '''
	#def __post_init__(self):
		# raise ValueError("app_name_to_suffx.keys not equals to app_names!")

	def app_names(self, case: case_mode = 'upper'):
		"""Application name list from app_name_to_stem_end dict keys"""
		return [k.strip().upper() if case == 'upper' else k.strip().lower() for k in self.app_name_to_stem_end.keys() if k.strip()]


	def app_names_as_enum(self, name='APP_NAME', module=__name__, case: case_mode='upper'):
		return Enum(name, self.app_names(case=case), module=module)

	app_name_to_stem_end: dict[str, str] = field(default_factory=lambda: {"TM":"jp.co.taimee", "MC":"jp.mercari.work.*"})
	"""('stem' means the part of the filename before the extension)Screenshot image filestem ends with value of this dict: filestem is the part of filename before its extention, delimited by underscore;BLOG pattern may be like: '*_{stem_end}.png' """
	# app_names: list[str] = field(default_factory=lambda: ['TM', 'MC'])

	stem_delimiter: str = "_"
	"""Delimiter in stem of filename: PREFIX_DATE_SUFFIX"""

	app: Optional[str] = None #: choices={', '.join(app_names())} 
	"""Application name of the screenshot to execute OCR"""

	image_dir: Optional[Path] = None # "~/Documents/screenshots"

	"""Image file root directory"""
	def app_name_enum(self, module=__name__)-> type[Enum]:
		return Enum('APP_NAME', self.app_names, module=module)

	image_ext_set: set[str] = field(default_factory=lambda:set([".png"]))
	"""Image file extension set, every extention starts with dot (default is {'.png'})"""

	shot_month: list[int] = field(default_factory=list)
	"""Choose Screenshot file by its month (MM part of [YYYY-MM-DD or YYYYMMDD]) included in filename stem. {Jan. is 01, Dec. is 12}(specified in a list like "[1,2,..]"""
	glob_pattern: str = "*.png"
	"""Image file name pattern as glob pattern to commit OCR or to get parameters."""
	glob_recursive: bool = True
	"""Recursive glob pattern matching"""
	files: list[str] = field(default_factory=list)
	"""Image file name list to commit OCR or to get parameters. Every file name's pattern is: <prefix>_<date>_<suffix>.<ext>"""

	image_area_param_section_stem: str = "image-area-param"
	"""Image area parameter section/table in image-area-param.ini"""
	app_border_ratio: dict[str, list[float]] = field( default_factory=lambda:{"taimee":[2.2,3.2]})
	"""Screenshot image file horizontal border ratio list of the app to execute OCR:(specified in format as "<app_name1>:<ratio1>,<ratio2> ..." )"""
	"""Screenshot image file name has suffix(sub extention) of the same as app name i.e. "<stem>.<suffix>.<ext>" (default: True)"""
	save: str = ''
	"""Output path to save OCR text of the image file as TOML format into the image file name extention as '.ocr-<app_name>.toml' """
	nth: int =1
	"""Rank(default: 1) of files descending sorted(the latest, the first) by modified date as wildcard(*, ?)"""
	glob_max: int = 60
	"""Pick up file max as glob pattern """
	app_suffix: bool = False
	"""Screenshot image file name has suffix(sub extention) of the same as app name i.e. "<stem>.<suffix>.<ext>" (default: True)"""
	show: bool = False
	"""Show images to check"""

	bin_image: bool = False
	"""Use binarized image for OCR"""
	no_ocr: bool = False
	"""Do not execute OCR"""
	ocr_conf: int = 55
	"""Confidence threshold for OCR"""
	psm: int = 6
	"""PSM value for Tesseract"""
	area_param_dir: str = ''
	"""Screenshot image area parameter config file directory"""
	area_param_name_list: list[str] = field(default_factory=area_param_names)
	"""Screenshot image area parameter name list"""
	area_param_file: str = "image-area-param.ini"
	"""Screenshot image area parameter config file: format as INI or TOML(".ini" or ".toml" extention respectively): in [image_area_param.<app>] section, items as "<area_name>=[<p1>,<p2>,<p3>,<p4>]" (e.g. "heading=[0,106,196,-1]") """
	ocr_filter_sqlite_db_name: str = "ocr-filter.db"
	"""SQLite DB file is created under `image-dir`/{yyyy} directory(yyyy is like 2025)"""
	data_year: int = 0
	"""Year of data (like -1, 0, 2025, ...). 0 means current year, negative value is difference from current year (like -1 means last year), positive value means a.d. year number (like 2025). If this value is larger than current year, an exception might be raised."""
	data_month: int = 0
	"""Month of data (like -1, 0, 1, 2, ...). 0 means current month, negative value is difference from current month (like -1 means last month), positive value means month number (1: Jan, 2: Feb, ...). If this value is larger than current month, data's date is treated as the last year."""
	show_ocr_area: bool = False
	"""Show every area before to commit OCR"""
	exclude_area_param_set: set[str] = field(default_factory=set) # { {f'{n}' for n in image_area_param_names()} }
	"""Exclude a set of image area parameter names"""

def append_doc(fd):
	return f"{fd}:{fd.default_factory()}"
MainSettings.__doc__ = MainSettings.__doc__ or '' + "\n".join([append_doc(fd) for fd in fields(MainSettings) if callable(fd.default_factory)])
def main_settings_from_dict(toml_dict: dict[str, Any]) -> MainSettings:
	return Binder(MainSettings).bind(toml_dict)
def main_settings_toml_lines()-> Iterator[str]:
	from dataclass_binder import Binder
	for line in Binder(MainSettings()).format_toml_template(): # Need to generate an instance to get default values of default factory
		yield(line)
def get_toml_path(fullpath: str|None, replacement_chars: str | None = "_-")-> Path:
	"""Get the default TOML filename from the given filename(as fullpath: <dir>/<stem>.<ext>). 
	If replacement_chars is provided as a sequence of 2 characters, replaces the 1st char with the 2nd char; 
	if replacement_chars is empty or None, no replacement happens."""
	node = Path(fullpath) if fullpath else Path(__file__)
	if not replacement_chars:
		toml = node.parent / (node.stem + '.toml')
	elif len(replacement_chars) >= 2:
		toml = node.parent / (node.stem.replace(replacement_chars[0], replacement_chars[1]) + '.toml')
	else:
		raise ValueError("Not enough replacement characters")
	if not toml.exists():
		logger.error("No proper TOML configuration file found at: %s", toml)
		raise FileNotFoundError(f"No proper TOML configuration file found at: {toml}")
	return toml
def load_main_settings(fullpath: Path, table: str = "")-> MainSettings:
	"""Load the TOML file and return the MainSettings instance of Binder"""
	with fullpath.open("rb") as f:
		config = tomllib.load(f)
	main_settings = Binder(MainSettings).bind(config[table] if table else config)
	return main_settings

from sys import stdout
def print_toml_template(settings: type[MainSettings]=MainSettings, as_class=False, out=stdout):
		for t in Binder(settings if as_class else settings()).format_toml_template():
			print(t, file=out)

if __name__ == '__main__':
	#print(MainSettings.__doc__)
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('--print-template', action='store_false', help='print TOML template of MainSettings')
	parser.add_argument('--as-class', action='store_false', help='print TOML template of MainSettings as class')
	args = parser.parse_args()
	if args.as_class:
		print("=== MainSettings as class ===")
		for t in Binder(MainSettings).format_toml_template():
			print(t)
	else:
		print("=== MainSettings as instance ===")
		for t in Binder(MainSettings()).format_toml_template():
			print(t)

