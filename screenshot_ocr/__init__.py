# my_package/__init__.py
import logging
import logging.handlers
from pathlib import Path
from sys import stderr
from datetime import datetime

LOGGING_NAME = Path(__file__).parent.stem
# Generate a filename like '2024-03-27.log'
log_filename = datetime.now().strftime(f"{LOGGING_NAME}_%Y-%m-%d.log")
logger = logging.getLogger(LOGGING_NAME)#.getChild(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(stderr)
console_handler.setLevel(logging.WARNING)
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-5s - [%(pathname)s:%(lineno)d] %(funcName)s:%(message)s'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.handlers.RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=5, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)