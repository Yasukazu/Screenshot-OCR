from . import logger as _logger
from argparse import ArgumentParser
# import logging
# from sys import path as sys_path
import re
from pathlib import Path
from enum import StrEnum, Enum
from simple_parsing import ArgumentParser as SimpleArgumentParser
from simple_parsing import parse, parse_known_args
# sys_path.insert(0, str(Path(__file__).parent))
# from edge_detect.image_filter_main_settings import print_toml_template
# dir_name = Path(__file__).parent.stem
logger = _logger.getChild(__name__) # logging.getLogger(dir_name)
# from set_logger import set_logger
# logger = set_logger(__name__)
MAIN_SETTINGS_PATH_STR = "SCREENSHOT_OCR_MAIN_SETTINGS_PATH"
MAIN_SETTINGS_PATH_DEFAULT = "main-settings.yaml"

class PrintSettingsExit(Exception):
	pass

class EmptyPathError(Exception):
	pass
class PathNotFoundError(Exception):
	pass

from sys import exit as sys_exit

def main():
	try:
		_main()
	except EmptyPathError:
		logger.error("Main settings path is empty")
		sys_exit(1)
	except PrintSettingsExit:
		sys_exit(0)

def _main():
	''' from sys import argv
	if len(argv) < 2:
		argv += ['-h'] '''
	parser = ArgumentParser(epilog="=== End of help ===", prog="screenshot-ocr", description=f"Screenshot OCR program: configuration file(in TOML format) fullpath is set by environment variable {MAIN_SETTINGS_PATH_STR}, or use default {MAIN_SETTINGS_PATH_DEFAULT} ", usage='%(prog)s [options]')
	parser.add_argument('-s', '--help-settings', action='store_true', help='print Settings options')
	parser.add_argument('-t', '--print-settings-template', help='print json or yaml format of Settings class', choices=['json', 'yaml']) # action='store_true',  
	parser.add_argument('-f', '--settings-file', type=Path, help=f'fullpath of Settings class in json or yaml format, while extention should be .json or .yaml respectively. Default is set by environment variable "{MAIN_SETTINGS_PATH_STR}" or `{MAIN_SETTINGS_PATH_DEFAULT}` in current directory.')

	args, unknown_args = parser.parse_known_args()
	from edge_detect.image_filter_main_settings import MainSettings
	if args.help_settings:
		try:
			parse(MainSettings, argv=["--help"])
		except SystemExit:
			raise PrintSettingsExit()
	elif args.print_settings_template:
			match args.print_settings_template:
				case 'json':
					print(MainSettings().dumps_json())
					raise PrintSettingsExit()
				case 'yaml':
					print(MainSettings().dumps_yaml())
					raise PrintSettingsExit()
				case _:
					pass
	from os import environ
	try:
		main_settings_path = args.settings_file or Path(environ[MAIN_SETTINGS_PATH_STR])
		if not main_settings_path:
			raise EmptyPathError()
		logger.info("Using main settings path from environment variable %s as %s",MAIN_SETTINGS_PATH_STR, main_settings_path)
	except (KeyError, EmptyPathError) as err:
		main_settings_path = Path().cwd() / MAIN_SETTINGS_PATH_DEFAULT
		logger.info("Using default main settings path %s since %s", main_settings_path, err)
	if not main_settings_path.exists():
		logger.warning("MainSettings config file %s does not exist; using default settings", main_settings_path)
		# sys_exit(1)
		main_settings = MainSettings()
	else:
		if main_settings_path.parts[0] == '~':
			main_settings_path = main_settings_path.expanduser()
		match main_settings_path.suffix.lower():
			case '.yaml' | '.yml' :
				#| '.json' | '.jsn':
				from yaml import safe_load
				with main_settings_path.open('r') as f:
					main_settings_dict = safe_load(f)
				main_settings_dict = transform_keys(main_settings_dict)
				main_settings = MainSettings(**main_settings_dict)
				#| '.json' | '.jsn':
				# main_settings = MainSettings.load(main_settings_path)
			case '.json' | '.jsn':
				from json import load as json_load
				with main_settings_path.open('r') as f:
					main_settings_dict = json_load(f)
				main_settings = MainSettings(**main_settings_dict)
			case _:
				raise ValueError(f"Unsupported file extension: {main_settings_path.suffix}")
		logger.info("Main settings loaded from %s: %s", main_settings_path, main_settings)
	# from simple_parsing import parse_known_args
	# main_settings, unknown_args = parse_known_args(MainSettings) # config_path=main_settings_path)
	# logger.info("Main settings loaded: %s", main_settings)
	# from simple_parsing import parse_known_args as simple_parse_known_args
	# s_args, s_unknown_args = simple_parse_known_args(MainSettings)
	'''arg_parser = SimpleArgumentParser()
	arg_parser.add_arguments(MainSettings, dest="main_settings")
	if args.help_settings:
		arg_parser.print_help()
		sys_exit(0)'''
	s_args, s_unknown_args = parse_known_args(MainSettings)
	logger.info("Arguments parsed: %s", s_args)
	cmd_opts = set([o.strip('-') for o in unknown_args if o.startswith('--')])
	for k, v in vars(s_args).items():
		if k in cmd_opts:
			setattr(main_settings, k, v)
			logger.info("  %s: %s (overridden by command line)", k, v)
	APP_NAMES = StrEnum('APP_NAMES', {k.upper(): v for k, v in main_settings.app_name_to_stem_end.items() if v and k})
	# APP_NAME = StrEnum('APP_NAME', [k.upper() for k in main_settings.app_name_to_stem_end.keys()])
	try:
		app_name_list = [getattr(APP_NAMES, main_settings.app.upper())] if main_settings.app else list(APP_NAMES)
	except KeyError as e:
		logger.error("'app' is not in `app_name_to_stem_end`: %s", e)
		sys_exit(1)
	logger.info("Available app names: %s", app_name_list)	
	if main_settings.output_dir and main_settings.output_dir[0] == '~':
		output_dir = Path(main_settings.output_dir).expanduser()
	else:
		output_dir = None
	for name in app_name_list:
		logger.info("processing %s", name)
		stem_end = name.value.strip(main_settings.stem_delimiter)
		for ext in main_settings.image_ext_set:
			if (_ext := ext.strip('.')):
				wildcard = "**/*" if main_settings.glob_recursive else "*"
				pattern = f"{wildcard}{main_settings.stem_delimiter}{stem_end}.{_ext}"
				for path in Path(main_settings.image_dir).expanduser().glob(pattern):
					logger.info("  found: %s", path)
					if output_dir:
						exec_ocr(path, output_dir)
from subprocess import run
def exec_ocr(input_file: Path, output_dir: Path, visualize: bool = True):
	'''Execute ndlocr-lite command'''
	if not input_file.exists():
		logger.error("input-file does not exist: %s", input_file)
		raise PathNotFoundError(f"input-file does not exist: {input_file}")
	output_dir.mkdir(parents=True, exist_ok=True)
	run(['ndlocr-lite', '--sourceimg', str(input_file), '--output', str(output_dir)] + (['--viz', 'True'] if visualize else []))
def kebab_to_snake(key):
	"""Converts a string from kebab-case to snake_case."""
	return re.sub(r'-', '_', key)

def transform_keys(data):
	"""Recursively transforms dictionary keys."""
	if isinstance(data, dict):
		return {kebab_to_snake(k): transform_keys(v) for k, v in data.items()}
	elif isinstance(data, list):
		return [transform_keys(i) for i in data]
	return data
if __name__ == '__main__':
	main()