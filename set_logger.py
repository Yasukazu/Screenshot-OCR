import logging
import logging.handlers
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
DOTENV_FILE = '.env'
found_dotenv = find_dotenv(DOTENV_FILE)
DOTENV_LOADED = load_dotenv(found_dotenv)
from os import environ
LOG_NAME = 'APP_LOG'
try:
	LOG_FILE = environ['APP_LOG']
except KeyError:
	LOG_FILE = LOG_NAME.replace('_', '.').lower()

def set_logger(name: str = __name__, debug=False, log_file_fullpath=LOG_FILE) -> logging.Logger:#~/logs/app.log
	"""
	Set up a logger with both console and file handlers.
	
	Args:
		name: Logger name (default: __name__)
		debug: Whether to enable debug level logging
		log_file_fullpath: Full path to log file (e.g., "~/logs/app.log")
	
	Returns:
		Configured logger instance
	"""
	formatter = logging.Formatter(
		'%(asctime)s| %(levelname)-5s | %(name)s.%(funcName)s.%(lineno)d | %(message)s'
	)
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)

	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	console_handler.setLevel(logging.DEBUG if debug or os.environ.get("DEBUG") == "1" else logging.WARNING)
	logger.addHandler(console_handler)

	if log_file_fullpath:
		log_file_path = Path(log_file_fullpath).expanduser()
		if not log_file_path.stem:
			raise ValueError("Error: log_file_fullpath must have a filename!")
		log_file_path.parent.mkdir(exist_ok=True)
		fileHandler = logging.handlers.RotatingFileHandler(
			log_file_path, maxBytes=1000000, backupCount=5, encoding="utf-8"
		)
		fileHandler.setFormatter(formatter)
		fileHandler.setLevel(logging.DEBUG)
		logger.addHandler(fileHandler)

	return logger

