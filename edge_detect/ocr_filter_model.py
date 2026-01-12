from pathlib import Path
from sys import path as sys_path
from sys import exit as sys_exit
from dotenv import load_dotenv

cwd = Path(__file__).resolve().parent
sys_path.insert(0, str(cwd.parent))
from set_logger import set_logger
logger = set_logger(__name__)
try:
	with (cwd / "ocr-filter.env").open() as envf:
		load_dotenv(stream=envf)
except FileNotFoundError as e:
	logger.error(f"Failed to load environment variables: {e}")
	sys_exit(1)
from os import environ
try:
	OCR_FILTER_DATA_YEAR = environ["OCR_FILTER_DATA_YEAR"]
	OCR_FILTER_DATA_MONTH = environ["OCR_FILTER_DATA_MONTH"]
	OCR_FILTER_SQLITE_DB_PATH = Path(environ["OCR_FILTER_SQLITE_DB_PATH"])
except (KeyError, ValueError) as e:
	logger.error(f"Failed to load environment variables: {e}")
	sys_exit(2)
except (TypeError) as e:
	logger.error(f"Failed to load environment variables: {e}")
	sys_exit(3)

if not OCR_FILTER_SQLITE_DB_PATH.exists():
	logger.info("OCR_FILTER_SQLITE_DB_PATH is not set")

from peewee import Model, SqliteDatabase, IntegerField, TextField, BlobField, BareField, CompositeKey, ForeignKeyField

database = SqliteDatabase(OCR_FILTER_SQLITE_DB_PATH)

class UnknownField(object):
	def __init__(self, *_, **__): pass

class BaseModel(Model):
	class Meta:
		database = database

class ImageFile(BaseModel):
	dir = TextField()
	class Meta:
		table_name = 'image_file_dir'

class PaystubOCR(BaseModel):
	app = IntegerField(null=True)
	day = IntegerField(null=True)
	image_file_dir = ForeignKeyField(ImageFile)
	image_file_name = TextField(null=True)
	checksum = TextField(null=True, unique=True)
	title = TextField(null=True)
	heading_text = TextField(null=True)
	shift_text = TextField(null=True)
	breaktime_text = TextField(null=True)
	paystub_text = TextField(null=True)
	salary_text = TextField(null=True)
	txt_lines = BareField(null=True)
	wages = IntegerField(null=True)

	class Meta:
		table_name = '-'.join(['paystub', OCR_FILTER_DATA_MONTH])
		indexes = (
			(('app', 'day'), True),
		)
		primary_key = CompositeKey('app', 'day')


if __name__ == "__main__":
	database.connect()
	database.create_tables([PaystubOCR, ImageFile])
	database.close()