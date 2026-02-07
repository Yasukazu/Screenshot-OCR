from enum import Enum
from tomllib import load as load_toml
from sys import stderr

from dotenv import dotenv_values

IMAGE_FILTER_MAIN_SETTINGS_FILE = 'IMAGE_FILTER_MAIN_SETTINGS_FILE'

def get_enum(enum_name: str, key_name: str, config_file: str = IMAGE_FILTER_MAIN_SETTINGS_FILE) -> Enum:
	try:
		main_config_file = dotenv_values()[config_file]
		with open(main_config_file, 'rb') as f:
			main_config = load_toml(f)
		return Enum(enum_name, main_config[key_name])
	except (TypeError, KeyError) as e:
		print(f"Not set {IMAGE_FILTER_MAIN_SETTINGS_FILE}: {e}", file=stderr)
		raise ValueError(f"Not set {IMAGE_FILTER_MAIN_SETTINGS_FILE}: {e}") from e
	except FileNotFoundError as e:
		print(f"Not found {IMAGE_FILTER_MAIN_SETTINGS_FILE}: {e}", file=stderr)
		raise ValueError(f"Not found {IMAGE_FILTER_MAIN_SETTINGS_FILE}: {e}") from e
	except Exception as e:
		print(f"Error: {e}", file=stderr)
		raise ValueError(f"Error: {e}") from e