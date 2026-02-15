from enum import Enum
from set_logger import set_logger
logger = set_logger(__name__)
from image_filter_main_settings import load_main_settings
try:
	main_settings = load_main_settings()
except Exception as e:
	logger.error("Failed to load main settings: %s", e)
	raise
APP_NAME = Enum('APP_NAME', main_settings.app_names)