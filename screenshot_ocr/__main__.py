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
	MAIN_SETTINGS_PATH_DEFAULT = "main-settings.toml"
	parser = ArgumentParser(epilog="=== End of help ===", prog="screenshot-ocr", description=f"Screenshot OCR program: configuration file(in TOML format) fullpath is set by environment variable {MAIN_SETTINGS_PATH_STR}, or use default {MAIN_SETTINGS_PATH_DEFAULT} "	)
	parser.add_argument('--print-template', action='store_true', help='print TOML template of MainSettings; remove leading "#" to specify any item')
	parser.add_argument('--as-class', action='store_true', help='print TOML template of MainSettings as class')
	parser.add_argument('--help-settings', action='store_true', help='print descriptions of fields in Settings class')

	args, unknown_args = parser.parse_known_args()
	from edge_detect.image_filter_main_settings import MainSettings
	if args.print_template:
		print_toml_template(MainSettings, as_class=args.as_class)
		sys_exit(0)
	from os import environ
	class EmptyPathError(Exception):
		pass
	try:
		main_settings_path = environ[MAIN_SETTINGS_PATH_STR]
		if not main_settings_path:
			raise EmptyPathError()
		logger.info("Using main settings path from environment variable %s as %s",MAIN_SETTINGS_PATH_STR, main_settings_path)
	except (KeyError, EmptyPathError) as err:
		main_settings_path = MAIN_SETTINGS_PATH_DEFAULT
		logger.info("Using default main settings path %s since %s", main_settings_path, err)
	from dataclass_binder import Binder
	main_settings = Binder(MainSettings).parse_toml(main_settings_path)
	logger.info("Main settings loaded: %s", main_settings)
	from simple_parsing import ArgumentParser as SimpleArgumentParser
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
	