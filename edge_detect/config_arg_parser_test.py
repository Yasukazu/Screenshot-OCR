from configparser import ConfigParser
from configargparse import ArgParser, TomlConfigParser, IniConfigParser
from os.path import abspath, dirname, join as os_path_join
from typing import Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


from image_filter import APP_NAME, ConfigError

def main(base_dir = abspath(dirname(__file__)), ini_file_name = "image-filter.ini"):
	ini_fullpath = os_path_join(base_dir, ini_file_name)
	arg_parser = ArgParser(default_config_files=[ini_fullpath], config_file_parser_class=IniConfigParser(['DEFAULT','stem_end'], split_ml_text_to_list=True))
	arg_parser.add_argument('--config', type=str, is_config_file=True, help='Config file path')
	config_parser = ConfigParser()
	# base_config: dict[str, Any] | None = None
	try:
		config_parser.read(ini_fullpath)
	except FileNotFoundError:
		logger.error("config file not found: %s", ini_fullpath)
		raise ConfigError("config file not found: %s" % ini_fullpath)
	except Exception as e:
		logger.error("config file load error: %s", ini_fullpath)
		raise ConfigError("config file load error: %s" % ini_fullpath)
	else:
		try:
			common_sect = config_parser["common"]
			image_ext = common_sect.get("image_ext", '')
			image_dir_base = common_sect.get("image_dir_base", '')
		except KeyError:
			image_ext = image_dir_base = ''
		try:
			app_stem_end = dict(config_parser['stem_end'])
		except KeyError:
			app_stem_end = {}

	from taimee_filter import TaimeeFilter
	OCR_FILTER = "ocr-filter"
	parser = ArgParser(default_config_files=[os_path_join(base_dir, "image-filter.toml")], config_file_parser_class=TomlConfigParser)
	parser.add_argument('files', nargs='*', help='Image files to commit OCR or to get parameters. Specify like: *.png')
	parser.add_argument('--app', choices=[n.name.lower() for n in APP_NAME], type=str, help=f'Application name of the screenshot to execute OCR:(specify in TOML filename =: {[f"*{n}{image_ext}" for n in app_stem_end]})') # 
	parser.add_argument('--toml', help=f'Configuration toml file name like {OCR_FILTER}')
	parser.add_argument('--save', help='Output path to save OCR text of the image file as TOML format into the image file name extention as ".ocr-<app_name>.toml"')
	parser.add_argument('--dir', help='Image dir of files: ./')
	parser.add_argument('--nth', type=int, default=1, help='Rank(default: 1) of files descending sorted(the latest, the first) by modified date as wildcard(*, ?)')
	parser.add_argument('--glob-max', type=int, default=60, help='Pick up file max as pattern found in TOML')
	parser.add_argument('--show', action='store_true', help='Show images to check')
	parser.add_argument('--make', action='store_true', help=f'make config. from image(i.e. this arg. makes not to load a config file like "{OCR_FILTER}.toml")')
	parser.add_argument('--no-ocr', action='store_true', default=False, help='Do not execute OCR')
	parser.add_argument('--ocr-conf', type=int, default=55, help='Confidence threshold for OCR')
	parser.add_argument('--psm', type=int, default=6, help='PSM value for Tesseract')
	parser.add_argument('--ini', default='image-filter.ini', help='Configuration ini file default:"image-filter.ini"')
	args = parser.parse_args()

if __name__ == '__main__':
	main()