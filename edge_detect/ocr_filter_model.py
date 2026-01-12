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
	OCR_FILTER_DATA_YEAR = int(environ["OCR_FILTER_DATA_YEAR"])
	OCR_FILTER_DATA_MONTH = int(environ["OCR_FILTER_DATA_MONTH"])
	OCR_FILTER_SQLITE_DB_PATH = Path(environ["OCR_FILTER_SQLITE_DB_DIR"]).expanduser() / environ["OCR_FILTER_SQLITE_DB_NAME"]
except KeyError as e:
	logger.error(f"Key error to load environment variables: {e}")
	sys_exit(2)
except ValueError as e:
	logger.error(f"Invalid value of environment variables: {e}")
	sys_exit(3)
except (TypeError) as e:
	logger.error(f"Invalid type(like None) to set value from environment variables: {e}")
	sys_exit(4)

if not OCR_FILTER_SQLITE_DB_PATH.exists():
	logger.info("OCR_FILTER_SQLITE_DB_PATH does not exist. It will be created.")

from peewee import Model, SqliteDatabase, IntegerField, TextField, BlobField, BareField, CompositeKey, ForeignKeyField

database = SqliteDatabase(OCR_FILTER_SQLITE_DB_PATH)

class UnknownField(object):
	def __init__(self, *_, **__): pass

class BaseModel(Model):
	class Meta:
		database = database

class ImageFile(BaseModel):
	root = TextField()
	# class Meta: table_name = 'image_file_dir'

class PaystubOCR(BaseModel):
	app = IntegerField()
	day = IntegerField()
	month = IntegerField()
	image_file_root = ForeignKeyField(ImageFile, backref='image_file_root')
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
		# table_name = '-'.join(['paystub', OCR_FILTER_DATA_MONTH])
		indexes = (
			(('app', 'day', 'month'), True),
		)
		primary_key = CompositeKey('app', 'day', 'month')


if __name__ == "__main__":
	database.connect()
	database.create_tables([PaystubOCR, ImageFile])
	database.close()