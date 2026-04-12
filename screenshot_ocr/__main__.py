from sys import path as sys_path
from pathlib import Path
sys_path.insert(0, str(Path(__file__).parent))
from edge_detect.image_filter_main_settings import print_toml_template
from set_logger import set_logger
logger = set_logger(__name__)
if __name__ == '__main__':
	from argparse import ArgumentParser
	from sys import exit as sys_exit
	''' from sys import argv
	if len(argv) < 2:
		argv += ['-h'] '''
	MAIN_SETTINGS_PATH_STR = "SCREENSHOT_OCR_MAIN_SETTINGS_PATH"
	MAIN_SETTINGS_PATH_DEFAULT = "main-settings.yaml"
	parser = ArgumentParser(epilog="=== End of help ===", prog="screenshot-ocr", description=f"Screenshot OCR program: configuration file(in TOML format) fullpath is set by environment variable {MAIN_SETTINGS_PATH_STR}, or use default {MAIN_SETTINGS_PATH_DEFAULT} "	)
	parser.add_argument('--help-settings', action='store_true', help='print Settings')
	parser.add_argument('--print-settings', choices=['json', 'yaml'], help='print into json/yaml format of Settings class')
	parser.add_argument('--settings-path', type=Path, help='fullpath of Settings class in json/yaml format')

	args, unknown_args = parser.parse_known_args()
	from edge_detect.image_filter_main_settings import MainSettings
	class PrintSettingsExit(Exception):
		pass
	try:
		match args.print_settings:
			case 'json':
				print(MainSettings().dumps_json())
				raise PrintSettingsExit()
			case 'yaml':
				print(MainSettings().dumps_yaml())
				raise PrintSettingsExit()
			case _:
				pass
	except PrintSettingsExit:
		sys_exit(0)
	from os import environ
	class EmptyPathError(Exception):
		pass
	try:
		main_settings_path = args.settings_path or Path(environ[MAIN_SETTINGS_PATH_STR])
		if not main_settings_path:
			raise EmptyPathError()
		logger.info("Using main settings path from environment variable %s as %s",MAIN_SETTINGS_PATH_STR, main_settings_path)
	except (KeyError, EmptyPathError) as err:
		main_settings_path = Path(MAIN_SETTINGS_PATH_DEFAULT)
		logger.info("Using default main settings path %s since %s", main_settings_path, err)
	if not main_settings_path.exists():
		logger.warning("MainSettings config file %s does not exist; using default settings", main_settings_path)
		# sys_exit(1)
		main_settings = MainSettings()
	else:
		main_settings = MainSettings.load(main_settings_path, drop_extra_fields=True)
		logger.info("Main settings loaded from %s: %s", main_settings_path, main_settings)
	# from dataclass_binder import Binder
	# main_settings = Binder(MainSettings).parse_toml(main_settings_path)
	# from simple_parsing import parse_known_args
	# main_settings, unknown_args = parse_known_args(MainSettings) # config_path=main_settings_path)
	# logger.info("Main settings loaded: %s", main_settings)
	from simple_parsing import ArgumentParser as SimpleArgumentParser
	# from simple_parsing import parse_known_args as simple_parse_known_args
	# s_args, s_unknown_args = simple_parse_known_args(MainSettings)
	arg_parser = SimpleArgumentParser()
	arg_parser.add_arguments(MainSettings, dest="main_settings")
	if args.help_settings:
		arg_parser.print_help()
		sys_exit(0)
	s_args, s_unknown_args = arg_parser.parse_known_args()
	logger.info("Arguments parsed: %s", s_args)
	unknown_opts = [o.strip('-') for o in unknown_args if o.startswith('--')]
	for k, v in vars(s_args.main_settings).items():
		if k in unknown_opts:
			setattr(main_settings, k, v)
			logger.info("  %s: %s (overridden by command line)", k, v)
	from enum import StrEnum
	from pathlib import Path
	APP_NAMES = StrEnum('APP_NAME', [main_settings.app.upper()] if main_settings.app else main_settings.app_names())
	logger.info("Available app names: %s", [name.name for name in APP_NAMES])	
	for name in APP_NAMES:
		logger.info("processing %s", name)
		stem_end = main_settings.app_name_to_stem_end[name.name].strip(main_settings.stem_delimiter)
		for ext in main_settings.image_ext_set:
			if (_ext := ext.strip('.')):
				wildcard = "**/" if main_settings.glob_recursive else ""
				blog_pattern = f"{wildcard}*{main_settings.stem_delimiter}{stem_end}.{_ext}"
				for path in Path(main_settings.image_dir).expanduser().glob(blog_pattern):
					logger.info("  found: %s", path)
