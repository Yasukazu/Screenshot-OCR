import logging
import logging.handlers
import os
from pathlib import Path


def set_logger(name: str | None = None, debug=False, log_file_fullpath="", loglevel=logging.INFO) -> logging.Logger:#~/logs/app.log
	"""
	Set up a logger with both console and file handlers.
	
	Args:
		name: Logger name
		debug: Whether to enable debug level logging
		log_file_fullpath: Full path to log file (e.g., "~/logs/app.log")
	
	Returns:
		Configured logger instance
	"""
	formatter = logging.Formatter(
		'%(asctime)s| %(levelname)-5s | %(name)s.%(funcName)s.%(lineno)d | %(message)s'
	)
	# logging.basicConfig( level=loglevel)
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	from sys import flags
	if flags.optimize > 0:  # -O or PYTHONOPTIMIZE>0
		loglevel = logging.WARNING
	
	streamHandler = logging.StreamHandler()
	streamHandler.setFormatter(formatter)
	streamHandler.setLevel(logging.DEBUG if debug or os.environ.get("DEBUG") == "1" else loglevel)
	logger.addHandler(streamHandler)

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

