from enum import Enum
from tomllib import load as load_toml
from dotenv import dotenv_values
IMAGE_FILTER_MAIN_SETTINGS_FILE = 'IMAGE_FILTER_MAIN_SETTINGS_FILE'
def get_app_name():
	try:
		main_config_file = dotenv_values()[IMAGE_FILTER_MAIN_SETTINGS_FILE]
		with open(main_config_file, 'rb') as f:
			main_config = load_toml(f)
		return Enum('APP_NAME', main_config['app-names'])
	except (TypeError, KeyError) as e:
		print(f"Not set {IMAGE_FILTER_MAIN_SETTINGS_FILE}: {e}")
		raise ValueError(f"Not set {IMAGE_FILTER_MAIN_SETTINGS_FILE}: {e}") from e
	except FileNotFoundError as e:
		print(f"Not found {IMAGE_FILTER_MAIN_SETTINGS_FILE}: {e}")
		raise ValueError(f"Not found {IMAGE_FILTER_MAIN_SETTINGS_FILE}: {e}") from e
	except Exception as e:
		print(f"Error: {e}")
		raise ValueError(f"Error: {e}") from e