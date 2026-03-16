""" Load num from env. val. """
from pathlib import Path
import sys
from dotenv import find_dotenv, dotenv_values
from dataclasses import dataclass
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
from set_logger import set_logger
import sys
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir) # Add to the beginning of the path
logger = set_logger(__name__)

ENV_PREFIX_STR = "IMAGE_FILTER"

@dataclass
class DotEnvInfo:
	filename: str
	path: str
	values: dict | None
	exception: Exception | None
	@property
	def is_valid(self) -> bool:
		return bool(self.path) and (self.exception is None)
	
DOTENV_INFO = DotEnvInfo(filename="", path="", values={}, exception=None)
DOTENV_INFO.filename = ".env"
DOTENV_INFO.values = None
DOTENV_INFO.path = find_dotenv(DOTENV_INFO.filename, raise_error_if_not_found=True)
DOTENV_INFO.exception = None
if DOTENV_INFO.path:
	logger.info("Loading .env file: %s", DOTENV_INFO.path)
	try:
		with open(DOTENV_INFO.path, 'r') as f:
			logger.info("Contents of .env file:\n%s", f.read())
			DOTENV_INFO.values = dotenv_values(DOTENV_INFO.path)
			logger.info("Loaded .env values: %s", dotenv_values)
		# load_dotenv(dotenv_path, override=True)
	except Exception as e:
		logger.warning("Failed to load .env file: %s", e)
		DOTENV_INFO.exception = e