# pyDAL (https://github.com/PyDAL/pydal)
from pathlib import Path
from datetime import datetime as Datetime
from sys import path as sys_path
from sys import exit as sys_exit
from typing import Sequence
from dotenv import load_dotenv
from pydal import DAL, Field

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
	OCR_FILTER_SQLITE_DB_PATH = Path(environ["OCR_FILTER_SQLITE_DB_DIR"]).expanduser() / environ["OCR_FILTER_DAL_DB_NAME"]
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
schm = f"sqlite://{environ['OCR_FILTER_DAL_DB_NAME']}"
db = DAL(schm,  folder=OCR_FILTER_SQLITE_DB_PATH.parent) # check_same_thread=False,

app_table = db.define_table('app', Field('id', 'integer', primary_key=True), Field('name', 'string', unique=True))
image_root_table = db.define_table('image_root',
	Field('id', 'integer', primary_key=True),
	Field('root', 'string', unique=True)
)
paystub_ocr_table = db.define_table('paystub_ocr',
	Field('id', 'integer', primary_key=True),
	Field('app', db.app), # foreign key
	Field('year', 'integer'),
	Field('month', 'integer'),
	Field('day', 'integer'),
	#modified_at = DateTimeField()
	# from_shift = DateTimeField(null=True)
	# to_shift = DateTimeField(null=True)
	# root = ForeignKeyField(ImageRoot, backref='paystub_ocr')
	Field('root', db.image_root),
	Field('file', 'string'),
	# image_file_name = TextField(null=True)
	# checksum = TextField(null=True, unique=True)
	Field('title', 'string'),
	Field('heading_text', 'string'),
	Field('shift_text', 'string'),
	Field('breaktime_text', 'string'),
	Field('paystub_text', 'string'),
	Field('salary_text', 'string'),
	# ocr_result = BareField(null=True)
	Field('wages', 'integer') 
	)

for name in APP_NAME:
	if not db.app(name=name):
		db.app.insert(name=name)
'''if not db.app(name=APP_NAME.MERCARI):
	app_id2 = db.app.insert(name=APP_NAME.MERCARI)'''
db.commit()
