""" MainSettings by DataclassBinder"""
from pathlib import Path
import tomllib
from typing import Any, Callable, Iterator
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from dataclass_binder import Binder
import sys
from simple_parsing import ArgumentParser
from deepdiff import DeepDiff
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
from set_logger import set_logger
import sys
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
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
@dataclass(kw_only=True)
class MainSettings:
	"""
	Extract/OCR paystub text from an image file: Files for OCR by 'files' option may be specified with app-name-suffix in wildcard(glob pattern matching like '--files *.<APP_NAME>*.png') or by 'shot-month' option (like '--shot_month -1' for last month, 0 for current month, other positive value for month number: Jan. is 1, Dec. is 12, ...) and 'app' option (like '--app taimee')
	"""

	app_name_to_suffix: dict[str, set[str]] = field(default_factory=lambda: {"taimee":{"co", "taimee"}, "mercari":{"mercari", "work"}})
	"""Screenshot image file suffix set: suffix is the part of filename before extention, delimiter is dot (.)"""
	app_names: list[str] = field(default_factory=lambda: ['TAIMEE', 'MERCARI'])
	"""Application name list"""
	app: str|None = None #: choices={', '.join(app_names())} 
	"""Application name of the screenshot to execute OCR"""
	def app_name_enum(self, module)-> type[Enum]:
		return Enum('APP_NAME', self.app_names, module=module)

	app_name_to_stem_end: dict[str, str] = field(default_factory=lambda: {'taimee': '_jp.co.taimee', 'mercari': '_jp.mercari.work.android'})
	"""Dictionary of 'app name' to 'stem end': 'stem' means the part of the filename before the extension"""

	image_ext_set: set[str] = field(default_factory=lambda:set([".png"]))
	"""Image file extension set, every extention starts with dot (default is {'.png'})"""
	image_dir: str = "~/Documents/screenshots"
	"""Image file root directory"""
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
	app_suffix: bool = False
	"""Screenshot image file name has suffix(sub extention) of the same as app name i.e. "<stem>.<suffix>.<ext>" (default: True)"""
	save: str = ''
	"""Output path to save OCR text of the image file as TOML format into the image file name extention as '.ocr-<app_name>.toml' """
	nth: int =1
	"""Rank(default: 1) of files descending sorted(the latest, the first) by modified date as wildcard(*, ?)"""
	glob_max: int = 60
	"""Pick up file max as pattern found in TOML"""
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
	"""SQLite DB file is created under `image_dir`/{yyyy} directory(yyyy is like 2025)"""
	data_year: int = 0
	"""Year of data (like -1, 0, 2025, ...). 0 means current year, negative value is difference from current year (like -1 means last year), positive value means a.d. year number (like 2025). If this value is larger than current year, an exception might be raised."""
	data_month: int = 0
	"""Month of data (like -1, 0, 1, 2, ...). 0 means current month, negative value is difference from current month (like -1 means last month), positive value means month number (1: Jan, 2: Feb, ...). If this value is larger than current month, data's date is treated as the last year."""
	show_ocr_area: bool = False
	"""Show every area before to commit OCR"""
	exclude_area_param_set: set[str] = field(default_factory=set) # { {f'{n}' for n in image_area_param_names()} }
	"""Exclude a set of image area parameter names"""

	@classmethod
	def from_dict(cls, toml_dict: dict[str, Any]) -> 'MainSettings':
		return Binder(MainSettings).bind(toml_dict)

def append_doc(fd):
	return f"{fd}:{fd.default_factory()}"
MainSettings.__doc__ = MainSettings.__doc__ or '' + "\n".join([append_doc(fd) for fd in fields(MainSettings) if callable(fd.default_factory)])
def main_settings_from_dict(toml_dict: dict[str, Any]) -> MainSettings:
	return Binder(MainSettings).bind(toml_dict)
def main_settings_toml_lines(Settings=MainSettings)-> Iterator[str]:
	from dataclass_binder import Binder
	for line in Binder(Settings()).format_toml_template(): # Need to generate an instance to get default values of default factory
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
def load_main_settings(fullpath: Path|str, settings_class=MainSettings, table: str = "") -> MainSettings:
	"""Load the TOML file and return the MainSettings instance of Binder"""
	with Path(fullpath).open("rb") as f:
		config = tomllib.load(f)
	main_settings = Binder(settings_class).bind(config[table] if table else config)
	return main_settings

def search_settings_file(script_fullpath: Path|str = Path(__file__), replace=('_', '-'))-> Path:
	"""Search for the TOML configuration file in the script directory and its parent directories"""
	if isinstance(script_fullpath, str):
		script_fullpath = Path(script_fullpath)
	script_dir = script_fullpath.expanduser().parent
	user_home_dir = Path('~').expanduser()
	main_settings_file = script_dir / (toml_name:=(Path(script_fullpath).stem.replace(replace[0], replace[1]) + '.toml'))
	found = False
	while not (found:=main_settings_file.exists()) and script_dir != user_home_dir:
		script_dir = script_dir.parent
		main_settings_file = script_dir / toml_name
	if found:
		return main_settings_file
	else:
		raise FileNotFoundError(f"No proper TOML configuration file found at: {main_settings_file}")
class SubSettings(MainSettings):
	pass
def load_merged_settings(main_settings_file: Path|str, main_settings_class = MainSettings, sub_settings_class = MainSettings, file_main_diff_list:list[str]|None=None, args_sub_diff_list:list[str]|None=None) -> MainSettings:
	"""Load and merge the main settings with the sub settings(descendent of main class: 'sub' is broader than 'main') from settings file(in TOML format, 'main' settings range) and command line parameters('sub' settings range)"""
	file_settings = load_main_settings(main_settings_file, main_settings_class)
	main_settings = main_settings_class()
	file_main_diff = DeepDiff(asdict(file_settings), (asdict(main_settings)))
	for key in file_main_diff.affected_root_keys:
		setattr(main_settings, key, getattr(file_settings, key))
		if file_main_diff_list is not None:
			file_main_diff_list.append(key)
	sub_settings = sub_settings_class()
	parser = ArgumentParser()
	parser.add_arguments(sub_settings_class, dest="settings") # from command line param.
	args = parser.parse_args()
	args_sub_diff = DeepDiff(asdict(args.settings), asdict(sub_settings))
	for key in args_sub_diff.affected_root_keys:
		setattr(sub_settings, key, getattr(args.settings, key))
		if args_sub_diff_list is not None:
			args_sub_diff_list.append(key)
	return sub_settings
def generate_toml_template(Settings=MainSettings, file=sys.stdout):
	for line in main_settings_toml_lines(Settings):
		print(line, file=file)
if __name__ == '__main__':
	#print(MainSettings.__doc__)
	print('-*-' * 20 + 'template'+ '-*-' * 20)
	print('\n'.join(main_settings_toml_lines()))
	print('-*-' * 20 + 'toml'+ '-*-' * 20)
	script_fullpath = Path(__file__)
	script_dir = script_fullpath.parent
	user_home_dir = Path('~').expanduser()
	main_settings_file = search_settings_file(__file__)#script_dir / (toml_name:=(script_fullpath.stem.replace('_', '-') + '.toml'))

	main_settings = load_main_settings(main_settings_file)
	merged_settings = load_merged_settings(main_settings_file)
	main_settings_dict = asdict(main_settings)
	merged_settings_dict = asdict(merged_settings)
	diff2 = DeepDiff(main_settings_dict, merged_settings_dict)
	for key in diff2.affected_root_keys:
		print(f"{key}: {getattr(merged_settings,key)}")
	# print(diff2)
	# print(f"{main_settings=}")
	# print('--- Arguments ---')
