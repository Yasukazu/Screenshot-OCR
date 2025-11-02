import logging
import logging.handlers
import os
from pathlib import Path


def set_logger(name: str | None = None, debug=False, log_file_fullpath=""):#~/logs/app.log"):
	formatter = logging.Formatter(
		'%(asctime)s| %(levelname)-5s | %(name)s.%(funcName)s.%(lineno)d | %(message)s'
	)
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)

	streamHandler = logging.StreamHandler()
	streamHandler.setFormatter(formatter)
	streamHandler.setLevel(logging.DEBUG if debug or os.environ.get("DEBUG") == "1" else logging.INFO)
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

