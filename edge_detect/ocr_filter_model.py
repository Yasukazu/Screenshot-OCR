from pathlib import Path
from datetime import datetime as Datetime
from sys import path as sys_path
from sys import exit as sys_exit
from os import environ
from typing import Sequence
from dotenv import load_dotenv

cwd = Path(__file__).resolve().parent
sys_path.insert(0, str(cwd.parent))
from set_logger import set_logger
logger = set_logger(__name__)
from edge_detect.image_filter import APP_NAME, ImageAreaParamName

'''try:
	with (cwd / "ocr-filter.env").open() as envf:
		load_dotenv(stream=envf)
except FileNotFoundError as e:
	logger.error(f"Failed to load environment variables: {e}")
	sys_exit(1)
try:
	# OCR_FILTER_DATA_YEAR = int(environ["OCR_FILTER_DATA_YEAR"])
	# OCR_FILTER_DATA_MONTH = int(environ["OCR_FILTER_DATA_MONTH"])
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
	logger.info("OCR_FILTER_SQLITE_DB_PATH does not exist. It will be created.")'''
from datetime import date as Date
from peewee import OperationalError, Model, SqliteDatabase, IntegerField, TextField, BlobField, BareField, CompositeKey, ForeignKeyField, DateField, DateTimeField
database_environ_str = 'OCR_FILTER_SQLITE_DB_PATH'
database = SqliteDatabase(environ[database_environ_str], pragmas={'foreign_keys': 1})

class UnknownField(object):
	def __init__(self, *_, **__): pass

class BaseModel(Model):
	class Meta:
		database = database

class App(BaseModel):
	name = TextField(unique=True)
	
class ImageRoot(BaseModel):
	root = TextField(unique=True)
	
'''class ImageFile(BaseModel):
	file = TextField(unique=True)'''

class PaystubOCR(BaseModel):
	app = ForeignKeyField(App, backref='paystub_ocr')
	year = IntegerField()
	month = IntegerField()
	day = IntegerField()
	modified_at = DateTimeField()
	from_shift = DateTimeField(null=True)
	to_shift = DateTimeField(null=True)
	root = ForeignKeyField(ImageRoot, backref='paystub_ocr')
	file = TextField()
	# image_file_name = TextField()
	checksum = TextField(null=True)
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
		'''indexes = (
			(('app', 'day', 'month'), True),
		)'''
		primary_key = CompositeKey('app', 'year', 'month', 'day')

def init(db_path: str=environ['OCR_FILTER_SQLITE_DB_PATH']):
	global database
	database = SqliteDatabase(db_path, pragmas={'foreign_keys': 1})
	database.connect()
	App.create_table(safe=True)
	ImageRoot.create_table(safe=True)
	PaystubOCR.create_table(safe=True)

# init()

def insert_ocr_data(app: APP_NAME, year: int, month: int, day: int, data: dict[ImageAreaParamName, str], file: Path, hours:Sequence[str]|None=None):
	if not database.is_connection_usable():
		database.connect()
	App.create_table(safe=True)
	ImageRoot.create_table(safe=True)
	PaystubOCR.create_table(safe=True)
	app_obj, created = App.get_or_create(name=app)
	if created:
		logger.info("Created app: %s as app_obj: %s", app, app_obj)
	resolved_root = str(file.parent.resolve()) 
	root_obj, created= ImageRoot.get_or_create(root=resolved_root)
	if created:
		logger.info("Created root: %s as root_obj: %s", resolved_root, root_obj)
	# except ImageRoot.DoesNotExist: root_model = ImageRoot.create(root=resolved_root)
	old_item = PaystubOCR.get_or_none(app==app_obj, year==year, month==month, day==day)
	if old_item is None: # if not old_item:
		checksum = get_file_checksum_md5(file)
		new_item = PaystubOCR.create(
			app=app_obj,
			year=year,
			month=month,
			day=day,
			modified_at=Datetime.now(),
			heading_text=data.get(ImageAreaParamName.HEADING),
			shift_text=data.get(ImageAreaParamName.SHIFT),
			breaktime_text=data.get(ImageAreaParamName.BREAKTIME),
			paystub_text=data.get(ImageAreaParamName.PAYSTUB),
			salary_text=data.get(ImageAreaParamName.SALARY),
			root=root_obj,
			file=file.name,
			checksum=checksum,
		)
		database.commit()
		return new_item
	# database.close()
import hashlib

def get_file_checksum_md5(filename, blocksize=65536):
	""" calculate md5 checksum of a file """
	hash_obj = hashlib.md5()
	with open(filename, "rb") as f:
		for block in iter(lambda: f.read(blocksize), b""):
			hash_obj.update(block)
	return hash_obj.hexdigest()

if __name__ == "__main__":
	if not database.is_connection_usable():
		database.connect()
	database.create_tables([PaystubOCR, ImageRoot, App])
	taimee = App.create(name='taimee')
	mercari = App.create(name='mercari')
	database.commit()
	database.close()
'''
query = ofm.PaystubOCR.select(ofm.PaystubOCR,ofm.App,ofm.ImageRoot).join(ofm.App).switch(ofm.PaystubOCR).join(ofm.ImageRoot)

In [21]: str(query)
Out[21]: 'SELECT "t1"."app_id", "t1"."year", "t1"."month", "t1"."day", "t1"."modified_at", "t1"."from_shift", "t1"."to_shift", "t1"."root_id", "t1"."file", "t1"."checksum", "t1"."title", "t1"."heading_text", "t1"."shift_text", "t1"."breaktime_text", "t1"."paystub_text", "t1"."salary_text", "t1"."ocr_result", "t1"."wages", "t2"."id", "t2"."name", "t3"."id", "t3"."root" FROM "paystubocr" AS "t1" INNER JOIN "app" AS "t2" ON ("t1"."app_id" = "t2"."id") INNER JOIN "imageroot" AS "t3" ON ("t1"."root_id" = "t3"."id")'
'''