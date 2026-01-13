from pathlib import Path
from datetime import datetime as Datetime
from sys import path as sys_path
from sys import exit as sys_exit
from typing import Sequence
from dotenv import load_dotenv

cwd = Path(__file__).resolve().parent
sys_path.insert(0, str(cwd.parent))
from set_logger import set_logger
logger = set_logger(__name__)
from image_filter import APP_NAME, ImageAreaParamName

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
from datetime import date as Date
from peewee import OperationalError, Model, SqliteDatabase, IntegerField, TextField, BlobField, BareField, CompositeKey, ForeignKeyField, DateField, DateTimeField

database = SqliteDatabase(OCR_FILTER_SQLITE_DB_PATH, pragmas={'foreign_keys': 1})

class UnknownField(object):
	def __init__(self, *_, **__): pass

class BaseModel(Model):
	class Meta:
		database = database

class App(BaseModel):
	name = TextField(unique=True)
	
class ImageRoot(BaseModel):
	root = TextField(unique=True)
	
class ImageFile(BaseModel):
	file = TextField(unique=True)
	# class Meta: table_name = 'image_file_dir'

class PaystubOCR(BaseModel):
	app = ForeignKeyField(App, backref='paystub_ocr')
	year = IntegerField()
	month = IntegerField()
	day = IntegerField()
	modified_at = DateTimeField()
	from_shift = DateTimeField(null=True)
	to_shift = DateTimeField(null=True)
	root = ForeignKeyField(ImageRoot, backref='paystub_ocr')
	file = ForeignKeyField(ImageFile, backref='paystub_ocr')
	image_file_name = TextField(null=True)
	checksum = TextField(null=True, unique=True)
	title = TextField(null=True)
	heading_text = TextField(null=True)
	shift_text = TextField(null=True)
	breaktime_text = TextField(null=True)
	paystub_text = TextField(null=True)
	salary_text = TextField(null=True)
	ocr_result = BareField(null=True)
	wages = IntegerField(null=True)

	class Meta:
		# table_name = '-'.join(['paystub', OCR_FILTER_DATA_MONTH])
		indexes = (
			(('app', 'day', 'month'), True),
		)
		primary_key = CompositeKey('app', 'year', 'month', 'day')

def insert_ocr_data(app: APP_NAME, year: int, month: int, day: int, data: dict[ImageAreaParamName, str], file: Path, hours:Sequence[str]|None=None):
	if not database.is_connection_usable():
		database.connect()
	app_model = {APP_NAME.TAIMEE : App.get(App.name=='taimee'),
	APP_NAME.MERCARI : App.get(App.name=='mercari')}[app]
	file_model = ImageFile.create(file=file.name)
	root_model = ImageRoot.create(root=file.parent.name)
	new_item = PaystubOCR.create(
		app=app_model,
		year=year,
		month=month,
		day=day,
		modified_at=Datetime.now(),
		heading_text=data.get(ImageAreaParamName.HEADING),
		shift_text=data.get(ImageAreaParamName.SHIFT),
		breaktime_text=data.get(ImageAreaParamName.BREAKTIME),
		paystub_text=data.get(ImageAreaParamName.PAYSTUB),
		salary_text=data.get(ImageAreaParamName.SALARY),
		root=root_model,
		file=file_model
	)
	database.commit()
	return new_item
	# database.close()

if __name__ == "__main__":
	if not database.is_connection_usable():
		database.connect()
	database.create_tables([PaystubOCR, ImageFile, ImageRoot, App])
	taimee = App.create(name='taimee')
	mercari = App.create(name='mercari')
	database.commit()
	database.close()