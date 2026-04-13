# my_package/__init__.py
import logging
from pathlib import Path
from sys import stderr

LOGGING_NAME = Path(__file__).parent.stem
logger = logging.getLogger(LOGGING_NAME)#.getChild(__name__)
logger.setLevel(logging.DEBUG)

# ハンドラ（出力先）の設定
console_handler = logging.StreamHandler(stderr)
console_handler.setLevel(logging.WARNING)
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s - %(name)s - %(module)s.%(funcName)s(): %(message)s'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOGGING_NAME + '.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)