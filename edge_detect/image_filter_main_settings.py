""" MainSettings by DataclassBinder"""
from pathlib import Path
import tomllib
from typing import Any, Callable, Iterator, Sequence, Type, Literal, get_args, TYPE_CHECKING
from dataclasses import asdict, dataclass, field, fields
from enum import Enum, StrEnum
#from dataclass_binder import Binder
import sys
from simple_parsing import ArgumentParser
from deepdiff import DeepDiff
from fancy_dataclass import version
from dotenv import find_dotenv, load_dotenv, dotenv_values
from os import environ as os_environ

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
from set_logger import set_logger
import sys
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
logger = set_logger(__name__)
from dot_env import DOTENV_INFO
if not DOTENV_INFO.is_valid:
	logger.error("Failed to load '.env' file.")
	raise ValueError("Failed to load '.env' file.")
else:
	if DOTENV_INFO.values is not None:
		os_environ.update(DOTENV_INFO.values)
		logger.info("Loaded .env values: %s", DOTENV_INFO.values)

MAIN_SETTINGS_FILENAME = os_environ.get("IMAGE_FILTER_MAIN_SETTINGS_FILENAME", "image-filter-main-settings.toml")
from app_to_suffix import make_app_to_suffix_strenum, AppToSuffix, AppName

def area_param_names():
	return ['HEADING', 'SHIFT', 'BREAKTIME', 'PAYSTUB', 'SALARY']
from fancy_dataclass import version
from tap import Tap
import typed_argparse as tap
#@dataclass
from app_to_suffix import make_app_to_suffix_strenum # APP_TO_SUFFIX
APP_TO_SUFFIX = make_app_to_suffix_strenum()
logger.info("APP_TO_SUFFIX: %s", list(APP_TO_SUFFIX))
APP_NAME = StrEnum('APP_NAME', [m.name for m in APP_TO_SUFFIX], module='__main__')
logger.info("APP_NAME: %s", list(APP_NAME))
class Settings(tap.TypedArgs):
	""" Base settings """
	app: APP_NAME | None = tap.arg(default=None, help="Application name to process OCR from its screenshots")
def settings_runner(settings: Settings):
	""" Run the settings """
	print(f"Running settings for app[{type(settings.app)}]: {settings.app=}")
# from tap import TapIgnore
#@version((1,6))
#@dataclass
TYPE_CHECKING = True
class AppSettings:
	""" Application_name to suffix mapping must be defined in the environment variable or in '.env' file as 'IMAGE_FILTER_APP_TO_SUFFIX=<app1>:<suffix1>,<app2>:<suffix2>,<app3>:<suffix3>' """
	if TYPE_CHECKING:
		app: AppName | None = None
	else:
		app: str | None = None
	""" Application name to process OCR from its screenshots """
	stem_delimiter: str = '_'
	""" Delimiter for splitting screenshot filename stem into 3 parts like:: prefix:'Screenshot', datetime:'yyyy-mm-ddThh:mm:ss', suffix:'com.example.app.name'"""
	app_to_suffix: AppToSuffix = AppToSuffix(make_app_to_suffix_strenum(make_strenum=False))
	""" Application name to suffix mapping """

	@property
	def app_name(self) -> Enum|None:
		if self.app is None:
			return None
		return APP_NAME[self.app]


	@classmethod
	def get_app_names(cls):
		"""Application name list"""
		return APP_NAMES

	@property
	def app_names(self) -> list[str]:
		"""Application name list"""
		return self.get_app_names()

	@classmethod
	def get_app_name_enum(cls, module: str = '__main__')-> type[Enum]:
		return Enum('APP_NAME', cls.get_app_names(), module=module)

	def app_name_to_suffix_set(self)-> dict[str, set[str]] :
		"""Screenshot image file suffix set: suffix is the part of file stem(filename before extention), stem delimiter is underscore (_)"""
		dic = {}
		for app_suffix in self.app_to_suffix:
			suffix = app_suffix.value
			dic[app_suffix.name] = set([s for s in suffix.split('.') if s])
		return dic
		

	@property
	def app_name_to_stem_end(self)-> dict[str, str]:
		"""Dictionary of 'app name' to 'stem end': 'stem' means the part of the filename before the extension"""
		return self.app_to_suffix.items()

GLOB_MODE_LITERAL = Literal['NONE', 'GLOB', 'RGLOB'] # Glob pattern 
GLOB_MODE = StrEnum('GLOB', GLOB_MODE_LITERAL.__args__)
APP_NAMES = [m.name for m in APP_TO_SUFFIX]
class AppBorderRatio:
	""" Application name to border ratio mapping """
	def __init__(self, s: str):
		"""Initialize from string like "APP1:0.1, 0.2;APP2:0.3, 0.4" while APPn must be in APP_NAMES """
		self.dic = {}
		for item in s.split(';'):
			if not item:
				break
			key, values = item.split(':')
			if key not in APP_NAMES:
				raise ValueError(f"Invalid app name: {key}")
			values = [v for v in values.split(',') if v]
			self.dic[key] = [float(v) for v in values]
	def __getitem__(self, key):
		""" Getter for obj[key] """
		return self.dic[key]
#@version((1,2,1))
#@dataclass(kw_only=True)
class MainSettings(AppSettings):
	"""
	Extract/OCR paystub text from an image file: Files for OCR by 'files' option may be specified with app-name-suffix in wildcard(glob pattern matching like '--files *.<APP_NAME>*.png') or by 'shot-month' option (like '--shot_month -1' for last month, 0 for current month, other positive value for month number: Jan. is 1, Dec. is 12, ...) and 'app' option (like '--app <APP_NAME>')
	"""

	image_ext_set: set[str] = set([".png"])
	"""Image file extension set, every extention starts with dot (default is {'.png'})"""
	image_dir: str = "~/Documents/screenshots"
	"""Image file root directory"""
	shot_months: list[int] = []
	"""Choose Screenshot file by its month (MM part of [YYYY-MM-DD or YYYYMMDD]) included in filename stem. {Jan. is 01, Dec. is 12}(specified in a list like "[1,2,..]"""
	glob: GLOB_MODE_LITERAL = GLOB_MODE.RGLOB.name
	"""Image file name pattern as glob pattern to commit OCR or to get parameters."""
	rglob: bool = True
	"""Search glob pattern matching Recursively in a directory tree downto every subdirectories"""
	recurse_symlinks: bool = False
	""" Use symbolic links for searching glob pattern"""
	case_sensitive: bool = False
	""" Segregate char case(capital/small) for searching glob pattern"""
	files: list[str] = []
	"""Image file name list to commit OCR or to get parameters. Every file name's pattern is: <prefix>_<date>_<suffix>.<ext>"""

	image_area_param_section_stem: str = "image-area-param"
	"""Image area parameter section/table in image-area-param.ini"""
	app_border_ratio: AppBorderRatio = AppBorderRatio('TAIMEE:2.2,3.2') #dict[str, list[float]] = field( default_factory=lambda:{"taimee":[2.2,3.2]})
	"""Screenshot image file horizontal border ratio list of the app to execute OCR:(specified in format as "<app_name1>:<ratio1>,<ratio2> ..." )"""
	app_suffix: bool = False
	"""Screenshot image file name has suffix(sub extention) of the same as app name i.e. "<stem>.<suffix>.<ext>" (default: True)"""
	save_as: str = ''
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
	area_param_name_list: list[str] = area_param_names()
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
	exclude_area_param_set: set[str] = set() # field(default_factory=set) # { {f'{n}' for n in image_area_param_names()} }
	"""Exclude a set of image area parameter names"""

	@classmethod
	def from_dict(cls, toml_dict: dict[str, Any]) -> 'MainSettings':
		return Binder(MainSettings).bind(toml_dict)

	@classmethod
	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		parent_doc = cls.__doc__ or ""
		child_doc = super(cls, cls).__doc__ or ""
		cls.__doc__ = parent_doc + "\n\n" + (child_doc or "")

	@classmethod
	def load_toml(cls, fullpath: Path|str, table: str = "") -> 'MainSettings':
		"""Load the TOML file and return the Settings instance of Binder"""
		with Path(fullpath).open("rb") as f:
			config = tomllib.load(f)
		main_settings = Binder(cls).bind(config[table] if table else config)
		return main_settings
	
	@classmethod
	def toml_lines(cls):
		""" Iterate TOML template lines"""
		return Binder(cls()).format_toml_template()
	
	def get_app_name(self):
		try:
			return self.app.upper()
		except AttributeError:
			return Path(self.glob).suffixes[-2].strip('.').upper()
	
	def glob_files(self):
		for ext in self.image_ext_set:
			glob_pattern = f"*.{self.app}*.{ext.strip('.')}"
			self.files += [str(p) for p in (Path(self.image_dir).rglob(glob_pattern, case_sensitive=self.case_sensitive, recurse_symlinks=self.recurse_symlinks) if self.rglob else Path(self.image_dir).glob(glob_pattern, case_sensitive=self.case_sensitive, recurse_symlinks=self.recurse_symlinks))]


def append_doc(fd):
	return f"{fd}:{fd.default_factory()}"

MainSettings.__doc__ = MainSettings.__doc__ or '' + "\n".join([append_doc(fd) for fd in fields(MainSettings) if callable(fd.default_factory)])

def main_settings_from_dict(toml_dict: dict[str, Any]) -> MainSettings:
	return Binder(MainSettings).bind(toml_dict)
def main_settings_toml_lines(Settings:Type[Settings]=MainSettings)-> Iterator[str]:
	"""Generate TOML lines from the given settings class"""
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
def load_main_settings(fullpath: Path|str, settings_class:Type[MainSettings]=MainSettings, table: str = "") -> MainSettings:
	"""Load the TOML file and return the Settings instance of Binder"""
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
	class SameDir(Exception):
		pass
	try:
		while not (found:=main_settings_file.exists()) :
			script_dir = script_dir.parent
			if script_dir == user_home_dir:
				raise SameDir()
			main_settings_file = script_dir / toml_name
		if found:
			return main_settings_file
	except SameDir:
		if (main_settings_file:=(script_dir / toml_name)).exists():
			return main_settings_file
	logger.error("No proper TOML configuration file found at: %s", main_settings_file)
	raise FileNotFoundError("No proper TOML configuration file found")

from deepmerge import always_merger
# result = always_merger.merge(base, next_dict)
def load_merged_settings(by_file_settings: MainSettings, main_settings_class = MainSettings, sub_settings_class = MainSettings) -> MainSettings:
	"""Load and merge the main settings with the sub settings(descendent of main class: 'sub' is broader than 'main') from settings file(in TOML format, 'main' settings range) and command line parameters('sub' settings range).
	Every difference in dict.value is replaced.
	Any difference in a list is appended.
	DeepDiff search results in keys:('file', 'args') are stored in dict. 'file_args_diff'."""

	from deepmerge import always_merger as am
	main_settings = main_settings_class()
	file_main_diff = DeepDiff((by_file_settings_dict:=asdict(by_file_settings)), (main_settings_dict:=asdict(main_settings)))

	main_merged_settings = am.merge(main_settings_dict, by_file_settings_dict)
	for key in file_main_diff.affected_root_keys:
		assert main_merged_settings[key] == by_file_settings_dict[key], f"Key {key} has different values: {main_merged_settings[key]} != {by_file_settings_dict[key]}"
	sub_settings = sub_settings_class()
	parser = ArgumentParser()
	parser.add_arguments(sub_settings_class, dest="settings") # from command line param.
	args = parser.parse_args()
	args_sub_diff = DeepDiff((args_settings_dict:=asdict(args.settings)), (sub_settings_dict:=asdict(sub_settings)))
	affected_args_sub_diff = {k:v for k,v in args_settings_dict.items() if k in args_sub_diff.affected_root_keys}
	affected_args_main_diff = {k:v for k,v in affected_args_sub_diff.items() if k in main_settings_dict.keys()}
	main_merged_settings = am.merge(main_merged_settings, affected_args_main_diff)
	main_merged_diff = {k:v for k,v in main_merged_settings.items() if k in file_main_diff.affected_root_keys or k in args_sub_diff.affected_root_keys}

	sub_merged_settings = am.merge(sub_settings_dict, args_settings_dict)
	for key in args_sub_diff.affected_root_keys:
		assert sub_merged_settings[key] == args_settings_dict[key], f"Key {key} has different values: {sub_merged_settings[key]} != {args_settings_dict[key]}"
	sub_merged_diff = {k:v for k,v in sub_merged_settings.items() if k in args_sub_diff.affected_root_keys}
	return sub_settings_class(** main_merged_diff, **sub_merged_diff)


def load_merged_settings_no_deep_merge(main_settings_file: Path|str, main_settings_class = MainSettings, sub_settings_class = MainSettings, file_main_diff_list:list[str]|None=None, args_sub_diff_list:list[str]|None=None) -> MainSettings:
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

from returns.result import safe
@safe
def load_main_settings_safely(file: str|Path = __file__, settings_class=MainSettings, replace: Sequence[str] = ('_', '-'), search_file=False) -> MainSettings:
	"""Load main settings(with exception handlings as messages: FileNotFoundError, OSError, tomllib.TOMLDecodeError, KeyError, ValueError) from a TOML file, the name is replaced the filename of the script as underscore(_) to hypen(-).
	Returns: (Settings, Path)"""
	try:
		toml_path = search_settings_file(file, replace) if search_file else Path(file).with_name(Path(file).stem.replace(replace[0], replace[1]) + '.toml')
		main_settings = settings_class.load_toml(toml_path)
	except FileNotFoundError as e:
		logger.error("TOML configuration file not found: %s", e)
		raise
	except OSError as e:
		logger.error("File system error accessing TOML configuration: %s", e)
		raise
	except tomllib.TOMLDecodeError as e:
		logger.error("TOML configuration file[%s] is malformed or contains invalid syntax: %s", toml_path, e)
		raise
	except (KeyError, ValueError) as e:
		logger.error("Configuration error in main settings: %s", e)
		raise
	except Exception as e:
		logger.error("Failed to load main settings: %s", e)
		raise
	else:
		logger.info("Loaded main settings: %s", main_settings)
		return main_settings

def print_toml_template(Settings:Type[Settings]=MainSettings, file=sys.stdout):
	""" print each line to the specified file """
	for line in main_settings_toml_lines(Settings):
		print(line, file=file)

if __name__ == '__main__':
	# for line in Binder(AppSettings).format_toml_template(): # Need to generate an instance to get default values of default factory print(line)
	from sys import argv
	parser = tap.Parser(Settings)
	import argcomplete
	#argcomplete.autocomplete(Settings)
	parser.bind(settings_runner).run()
	#app_settings = AppSettings().parse_args(argv[1:]) #config_files=['app-config.json']
	exit(0)
	main_settings = MainSettings(underscores_to_dashes=True).parse_args(argv[1:]) #config_files=['main-config.json']
	from simple_parsing import parse as simple_parse
	app_settings: AppSettings = simple_parse(config_class=AppSettings, config_path='app-config.yaml')#, add_config_path_arg
	app_settings.app_to_suffixes |= AppSettings().app_to_suffixes # add default values