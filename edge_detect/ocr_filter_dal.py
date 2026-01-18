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
	# OCR_FILTER_DATA_YEAR = int(environ["OCR_FILTER_DATA_YEAR"])
	# OCR_FILTER_DATA_MONTH = int(environ["OCR_FILTER_DATA_MONTH"])
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

def make_db(schm: str = f"sqlite://{environ['OCR_FILTER_DAL_DB_NAME']}", folder=environ["OCR_FILTER_SQLITE_DB_DIR"], check_same_thread=True) -> DAL:
	return DAL(schm=schm, folder=folder, check_same_thread=check_same_thread)

def get_tables(db: DAL = make_db()) -> tuple[DAL.Table, DAL.Table, DAL.Table]:
	try:
		app_table = db.app
	except AttributeError:
		app_table = db.define_table('app', Field('id', 'integer', primary_key=True), Field('name', 'string', unique=True))
	try:
		image_dir_table = db.image_dir
	except AttributeError:
		image_dir_table = db.define_table('image_dir',
		Field('id', 'integer', primary_key=True),
		Field('dir', 'string', unique=True)
	)
	try:
		paystub_ocr_table = db.paystub_ocr
	except AttributeError:
		paystub_ocr_table = db.define_table('paystub_ocr',
		# Field('id', 'integer', primary_key=True),
		Field('app', db.app), # foreign key
		Field('year', 'integer'),
		Field('month', 'integer'),
		Field('day', 'integer'),
		#modified_at = DateTimeField()
		# from_shift = DateTimeField(null=True)
		# to_shift = DateTimeField(null=True)
		# root = ForeignKeyField(ImageRoot, backref='schm=schm, folder=folder, check_same_thread=check_same_threadpaystub_ocr')
		Field('dir', db.image_dir),
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
		Field('wages', 'integer'),
		primarykey = ['app', 'year', 'month', 'day']
		)
	if not app_table or not image_dir_table or not paystub_ocr_table:
		raise ValueError("Failed to create tables")
	return app_table, image_dir_table, paystub_ocr_table


def insert_ocr_data(db: DAL, app: APP_NAME, year: int, month: int, day: int, data: dict[ImageAreaParamName, str], file: Path, hours:Sequence[str]|None=None) -> int | None:
	app_table, image_dir_table, paystub_ocr_table = get_tables(db)
	if not (app_id := app_table(name=app)):
		app_id = app_table.insert(name=app)
	dir = file.parent
	if not (dir_id := image_dir_table(dir=dir)):
		dir_id = image_dir_table.insert(dir=dir)
	item_id = paystub_ocr_table.update_or_insert(db.paystub_ocr.app == app_id & db.paystub_ocr.year == year & db.paystub_ocr.month == month & db.paystub_ocr.day == day,
		app=app_id,
		year=year,
		month=month,
		day=day,
		heading_text=data.get(ImageAreaParamName.HEADING),
		shift_text=data.get(ImageAreaParamName.SHIFT),
		breaktime_text=data.get(ImageAreaParamName.BREAKTIME),
		paystub_text=data.get(ImageAreaParamName.PAYSTUB),
		salary_text=data.get(ImageAreaParamName.SALARY),
		dir=dir_id,
		file=file.name
	)
	if item_id:
		db.commit()
		return item_id

if __name__ == '__main__':
	inserted = False
	db = make_db()
	app_table, image_dir_table, paystub_ocr_table = get_tables(make_db())
	for name in APP_NAME:
		if not db.app(name=name):
			db.app.insert(name=name)
			inserted = True
	'''if not db.app(name=APP_NAME.MERCARI):
		app_id2 = db.app.insert(name=APP_NAME.MERCARI)'''
	if inserted:
		db.commit()
