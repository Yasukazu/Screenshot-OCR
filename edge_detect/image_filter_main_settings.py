""" MainSettings by DataclassBinder"""
# PYTHON_ARGCOMPLETE_OK 
from pathlib import Path
import tomllib
import traceback
from typing import Annotated, Any, Callable, Iterator, Sequence, Type, Literal, get_args, TYPE_CHECKING, List
from dataclasses import asdict, dataclass, field, fields
from enum import Enum, StrEnum
#from dataclass_binder import Binder
import sys
# from simple_parsing import ArgumentParser
from deepdiff import DeepDiff
from os import environ as os_environ

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
from set_logger import set_logger
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
tapr_path = Path(__file__).resolve().parent.parent / 'tapr' # 'typed_argparser')
if tapr_path not in sys.path:
	sys.path.insert(0, str(tapr_path))
logger = set_logger(__name__)
from dot_env import DOTENV_INFO, ENV_PREFIX_STR
APP_TO_SUFFIX_KEY = "APP_TO_SUFFIX"
ENV_PREFIX_APP_TO_SUFFIX_KEY = '_'.join([ENV_PREFIX_STR, APP_TO_SUFFIX_KEY])
try:
	APP_TO_SUFFIX_VALUE = os_environ[ENV_PREFIX_APP_TO_SUFFIX_KEY]
except KeyError as e:
	try:
		APP_TO_SUFFIX_VALUE = DOTENV_INFO.values[ENV_PREFIX_APP_TO_SUFFIX_KEY]
	except (TypeError, KeyError) as err:
		logger.error("Failed to get '%s' from environment variable or '.env' file:%s", APP_TO_SUFFIX_KEY, err)
		raise err

# from app_to_suffix import make_app_to_suffix_strenum, AppToSuffix, AppName
# APP_TO_SUFFIX = make_app_to_suffix_strenum(strenum_name=APP_TO_SUFFIX_KEY, name_value_pair=APP_TO_SUFFIX_VALUE)
from app_to_suffix import APP_TO_SUFFIX
MAIN_SETTINGS_FILENAME = os_environ.get("IMAGE_FILTER_MAIN_SETTINGS_FILENAME", "image-filter-main-settings.toml")

AREA_PARAM_NAME = Enum('AREA_PARAM_NAME', ['HEADING', 'SHIFT', 'BREAKTIME', 'PAYSTUB', 'SALARY'])
logger.info("APP_TO_SUFFIX: %s", list(APP_TO_SUFFIX))
APP_NAME = StrEnum('APP_NAME', [m.name for m in APP_TO_SUFFIX], module='__main__')
logger.info("APP_NAME: %s", list(APP_NAME))

'''from typed_argparser import ArgumentClass, Field
from typed_argparser.validators import ArgumentValidator
from typed_argparser.exceptions import ValidationError # ArgumentError, ValidatorInitError, 
from typed_argparser.types import Args

class AppNameValidator(ArgumentValidator):
	choices = [m.name.lower() for m in APP_NAME]
	def __init__(self):
		self.choices = [m.name for m in APP_NAME]
	def validator(self, value: str) -> None:
		if value.lower() not in self.choices:
			raise ValidationError(f"Invalid app name: {value}")

class ChoicesValidator(ArgumentValidator):
	def __init__(self, choices: list[str]):
		self.choices = [choice for choice in choices]

	def validator(self, value: str) -> None:
		if value not in self.choices:
			raise ValidationError(f"Invalid choice:'{value}' in {self.choices}")
class KeyValidator(ArgumentValidator):
	def __init__(self, keys: list[str]):
		self.keys = [key for key in keys]

	def validator(self, value: dict[str, Any]) -> None:
		for key in value.keys():
			if key not in self.keys:
				raise ValidationError(f"Invalid key:'{value}' in {self.keys}") '''
# Create a constant for the glob choices to avoid type issues
GLOB_CHOICES = ['NONE', 'GLOB', 'RGLOB']
from app_sym import APP_SYM # app symbol to number mapping
def get_app_sym_to_suffix_from_env(env_file: str = ".env.screenshot_ocr", env_prefix="SCREENSHOT_OCR_", env_name="APP_SYM_TO_SUFFIX") -> dict[str, str]|None:
	""" Get the app symbol to suffix mapping from environment variables using dotenv(environment variables is supplemented with env_file)"""
	from dotenv import load_dotenv, find_dotenv
	load_dotenv(find_dotenv(env_file))
	try:
		screenshot_app_to_suffix = os_environ[f"{env_prefix}{env_name}"]
		if not screenshot_app_to_suffix:
			raise ValueError(f"Empty environment variable {env_prefix}{env_name}")
	except KeyError:
		raise ValueError(f"Environment variable {env_prefix}{env_name} not found")
	dic = {}
	app_sym_set = set([app_sym.name for app_sym in APP_SYM])
	for mapping in screenshot_app_to_suffix.split(','):
		try:
			k, v = mapping.split(':')
		except ValueError:
			continue
		else:
			if k not in app_sym_set:
				raise ValueError(f"Invalid app symbol: {k}")
			if not v:
				raise ValueError(f"Empty suffix for app symbol: {k}")
			dic[k] = v
	return dic

from typing import Optional
from pydantic import model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

class Settings(BaseSettings):
	""" Base settings """
	model_config = SettingsConfigDict(cli_parse_args=True, env_prefix='SCREENSHOT_OCR_', env_file=".env", env_file_encoding="utf-8", env_nested_delimiter='__', extra='ignore', toml_file="config.toml")
	app: Optional[APP_NAME] = Field( default=None,
		description=f"Application symbol to process OCR from its screenshots;symbols are defined in `app_sym.py`:{{{'|'.join([m.name for m in APP_SYM])}}}.")#, validator=ChoicesValidator([m.name for m in APP_SYM])) # default=None,ChoicesValidator.choices

	app_sym_to_suffix: dict[APP_NAME, str] = Field(default_factory=dict,
		description="Application symbol to suffix mapping") #, validator=KeyValidator([app_sym.name for app_sym in APP_SYM]))default=get_app_sym_to_suffix_from_env(),
	def model_post_init(self, __context):
		toml_file = self.model_config.get('toml_file', None)
		try:
			print(f"TOML file used: {Path(toml_file).absolute()}")
		except Exception as e:
			print(f"Error getting TOML file: {e}")
		else:
			print(f"TOML values loaded:")
			for field_name, field_info in self.__class__.model_fields.items():
				value = getattr(self, field_name)
				print(f"  {field_name}: {value}")
		print("\n=== FIELD → ENV MAPPING ===")
		for field_name, field in self.__class__.model_fields.items():
			env_name = field.alias or field_name.upper()
			value = getattr(self, field_name)
			print(f"{field_name:<15} → {env_name:<20} = {value}")
	@classmethod
	def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings, **kwargs):
		return (settings_cls, TomlConfigSettingsSource(settings_cls), env_settings, dotenv_settings, file_secret_settings)
	''' @property
	def app_suffix(self)-> str | None:
		""" Get the app's suffix(last part of file stem before '_') StrEnum """
		if self.app is None:
			return None
		return self.app_sym_to_suffix[self.app.upper()]

	@property
	def app_name(self)-> APP_SYM | None:
		""" Get the app's Enum type """
		if self.app is None:
			return None
		return APP_SYM[self.app.upper()] '''

def settings_runner(settings: Settings):
	""" Run the settings """
	print(f"Running settings for app[{type(settings.app)}]: {settings.app=}\napp_to_suffix[{type(settings.app_sym_to_suffix)}]: {settings.app_sym_to_suffix=}")

# from tap import TapIgnore
#@version((1,6))
#@dataclass
TYPE_CHECKING = True
class AppSettings(Settings):
	""" Application_name to suffix mapping must be defined in the environment variable or in '.env' file as 'IMAGE_FILTER_APP_TO_SUFFIX=<app1>:<suffix1>,<app2>:<suffix2>,<app3>:<suffix3>' """
	stem_delimiter: Optional[str] = Field(default='_', description= " Delimiter for splitting screenshot filename stem into 3 parts like:: prefix:'Screenshot', datetime:'yyyy-mm-ddThh:mm:ss', suffix:'com.example.app.name'")

	@classmethod
	def get_app_names(cls):
		"""Application name list"""
		return APP_NAMES

	@property
	def app_names(self) -> list[str]:
		"""Application name list"""
		return self.get_app_names()

	def app_name_to_suffix_set(self)-> dict[str, set[str]] :
		"""Screenshot image file suffix set: suffix is the part of file stem(filename before extention), stem delimiter is underscore (_)"""
		dic = {}
		for app_suffix in self.app_sym_to_suffix:
			suffix = app_suffix.value
			dic[app_suffix.name] = set([s for s in suffix.split('.') if s])
		return dic
		

	@property
	def app_name_to_stem_end(self)-> dict[str, str]:
		"""Dictionary of 'app name' to 'stem end': 'stem' means the part of the filename before the extension"""
		return self.app_suffix.items()

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
#@dataclass(kw_only=True)
class MainSettings(AppSettings):
	__program__ = "ocr-filter"
	__description__ = "Extract/OCR paystub text from an image file: Files for OCR by 'files' option may be specified with app-name-suffix in wildcard(glob pattern matching like '--files *.<APP_NAME>*.png') or by 'shot-month' option (like '--shot_month -1' for last month, 0 for current month, other positive value for month number: Jan. is 1, Dec. is 12, ...) and 'app' option (like '--app <APP_NAME>')"
	__version__ = "1.2.1"
	__usage__ = f"{__program__} --<option1> {{value1}} --<option2> {{value2}} ..."
	__epilog__ = "\n`MainSettings` ends.\n"
	

	image_ext_set: Optional[List[str]] = Field(default=".png",
		description="Image file extension set, every extention starts with dot (default is {'.png'})")
	image_dir: Optional[str] = Field(default="~/Documents/screenshots",
		description="Image file root directory")
	shot_months: list[int]|None = Field(
		description="Choose Screenshot file by its month (MM part of [YYYY-MM-DD or YYYYMMDD]) included in filename stem. {Jan. is 01, Dec. is 12}(specified in a list like '[1,2,..]')")
	glob: Optional[str] = Field(default="RGLOB",
		# validator=ChoicesValidator(GLOB_CHOICES),
		description=f"Glob mode, Image file name matching pattern by glob('*_{{app_suffix}}.{{ext}}') to commit OCR or to get parameters, choose from {GLOB_CHOICES};RGLOB: Recursively search in a directory tree downto every subdirectories")

	@property
	def is_rglob(self) -> bool:
		"""Search glob pattern matching Recursively in a directory tree downto every subdirectories"""
		return self.glob == "RGLOB"
	recurse_symlinks: Optional[bool] = Field(default=False,
			description="Use symbolic links for searching glob pattern")
	case_sensitive: Optional[bool] = Field(default=False,
		description="Segregate char case(capital/small) for searching glob pattern")
	files: Optional[list[str]] = Field(
		description="Image file name list to commit OCR or to get parameters. Every file name's pattern is: <prefix>_<date>_<suffix>.<ext>")
	image_area_param_section_stem: str = Field(default="image-area-param",
		description="Image area parameter section/table in image-area-param.ini")
	app_border_ratio: dict[str, str]|None = Field(default=None, #AppBorderRatio('TAIMEE:2.2,3.2') #dict[str, list[float]] = field( default_factory=lambda:{"taimee":[2.2,3.2]})
		description="Screenshot image file horizontal border ratio list of the app to execute OCR:(specified in format as '<app_name1>:<ratio1>,<ratio2> ...')")
	app_is_suffix: bool = Field(default=False,
			description="Screenshot image file name has suffix(sub extention) of the same as app name i.e. '<stem>.<suffix>.<ext>' (default: True)")
	save_dir: Optional[Path] = Field( # Args()
		description="Output file directory where to save OCR text of the image file in TOML format into the file with name as '<stem>.ocr-<app_name>.toml' while executing OCR")
	nth: int = Field(default=1,
		description="Rank(first, second, ...) of files descending sorted(the latest, the first) by modified datetime as wildcard(*, ?)")
	glob_max: int = Field(default=100,
		description="Pick up files max. count found in glob pattern")
	show: bool = Field(default=False,
		description="Show images to check")

	bin_image: bool = Field(default=False, description="Use binarized image for OCR")
	no_ocr: bool = Field(default=False, description="Do not execute OCR")
	ocr_conf: int = Field(default=55, description="Confidence threshold for OCR")
	psm: int = Field(default=6, description="PSM value for Tesseract")
	area_param_dir: Optional[Path] = Field(description="Screenshot image area parameter config file directory")
	area_param_name_list: list[str] = Field(default=[m.name for m in AREA_PARAM_NAME], description="Screenshot image area parameter name list")
	area_param_file: Path = Field(default="image-area-param.ini", description='Screenshot image area parameter config file: format as INI with ".ini" extention): in [image_area_param.<app>] section, items as "<area_name>=[<p1>,<p2>,<p3>,<p4>]" (e.g. "heading=[0,106,196,-1]") ')
	ocr_filter_sqlite_db_name: Path = Field(default="ocr-filter.db", description="SQLite DB file is going to be created in `image_dir` directory if not exists.")#(yyyy is like 2025)')
	data_year: int = Field(default=0, description='Year of data (like -1, 0, 2025, ...). 0 means current year, negative value is difference from current year (like -1 means last year), positive value means a.d. year number (like 2025). If this value is larger than current year, an exception might be raised.')
	data_month: int = Field(default=0, description="Month of data (like -1, 0, 1, 2, ...). 0 means current month, negative value is difference from current month (like -1 means last month), positive value means month number (1: Jan, 2: Feb, ...). If this value is larger than current month, data's date is treated as the last year.")
	show_ocr_area: bool = Field(default=False, description="Show every area before to commit OCR")
	exclude_area_param_set: Optional[set[str]] = Field(description='Exclude a set of image area parameter names')

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

# MainSettings.__doc__ = MainSettings.__doc__ or '' + "\n".join([append_doc(fd) for fd in fields(MainSettings) if callable(fd.default_factory)])

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

# from deepmerge import always_merger
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
	from sys import argv, exit
	try:
		settings = Settings()
	except Exception as e:
		traceback.print_exc()
		logger.error("Failed to load settings: %s", e)
		exit(1)
	exit(0)
	''' settings.parse(' '.join(argv[1:]))
	@settings.execute('app')
	def run1(app: str):
		print(f"Running settings for app[{app.__class__}]: settings.app={app}") '''
		#settings_runner(settings)
	#run()
	# from typed_argparse import Parser
	# parser = Parser(Settings, usage=f"%(prog)s [--app {{{'|'.join([n.lower() for n in APP_NAMES])}}}];Set env. variable: {ENV_PREFIX_APP_TO_SUFFIX_KEY}=<app_name1>:<app_suffix1>,<app_name2>:<app_suffix2>;Suffix is the last part of split-by-underscore('_') in the image file name's stem part (filename part before extention:last part of filename(like '.png')).", epilog="\nDone.")
	# import argcomplete
	# argcomplete.autocomplete(parser)
	# parser.bind(settings_runner).run()
	#app_settings = AppSettings().parse_args(argv[1:]) #config_files=['app-config.json']
	main_settings = MainSettings()#underscores_to_dashes=True).parse_args(argv[1:]) #config_files=['main-config.json']
	main_settings.parse(argv[1:])#' '.join(
	@main_settings.execute('app')
	def run2(app: str):
		print(f"Running main settings for app[{app.__class__}]: main_settings.app={app}")
		#main_settings_runner(main_settings)
	'''from simple_parsing import parse as simple_parse
	app_settings: AppSettings = simple_parse(config_class=AppSettings, config_path='app-config.yaml')#, add_config_path_arg
	app_settings.app_to_suffixes |= AppSettings().app_to_suffixes # add default values'''