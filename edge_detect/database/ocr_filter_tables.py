from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, MetaData, Table, Text
from sqlalchemy.sql.sqltypes import NullType

metadata = MetaData()


t_app = Table(
	'app', metadata,
	Column('id', Integer, primary_key=True),
	Column('name', Text, nullable=False),
	Index('app_name', 'name', unique=True)
)

t_imagefile = Table(
	'imagefile', metadata,
	Column('id', Integer, primary_key=True),
	Column('file', Text, nullable=False),
	Index('imagefile_file', 'file', unique=True)
)

t_imageroot = Table(
	'imageroot', metadata,
	Column('id', Integer, primary_key=True),
	Column('root', Text, nullable=False),
	Index('imageroot_root', 'root', unique=True)
)

t_paystubocr = Table(
	'paystubocr', metadata,
	Column('app_id', ForeignKey('app.id'), primary_key=True),
	Column('year', Integer, primary_key=True),
	Column('month', Integer, primary_key=True),
	Column('day', Integer, primary_key=True),
	Column('modified_at', DateTime, nullable=False),
	Column('from_shift', DateTime),
	Column('to_shift', DateTime),
	Column('root_id', ForeignKey('imageroot.id'), nullable=False),
	Column('file', Text, nullable=False),
	Column('checksum', Text),
	Column('title', Text),
	Column('heading_text', Text),
	Column('shift_text', Text),
	Column('breaktime_text', Text),
	Column('paystub_text', Text),
	Column('salary_text', Text),
	Column('ocr_result', NullType),
	Column('wages', Integer),
	Index('paystubocr_app_id', 'app_id'),
	Index('paystubocr_root_id', 'root_id')
)
if __name__ == '__main__':
	from os import environ
	from dataclasses import dataclass
	from dotenv import load_dotenv, find_dotenv
	# from configargparser import TypeArgumentParser
	from tap import Tap
	from sys import argv
	#@dataclass
	class ArgParser(Tap):
		find_env_file = True
		env_file: str = '.env' # environment variable file name

	args = ArgParser().parse_args()

	'''try:
		env_file = argv[1]
	except IndexError:
		env_file = None '''
	if not args.find_env_file:
		with open(args.env_file) as rf:
			load_dotenv(stream=rf)
	else:
		load_dotenv(find_dotenv(
		filename = args.env_file,
		raise_error_if_not_found = True,
		usecwd = True,
		) )
	from sqlalchemy import create_engine
	engine = create_engine('sqlite:+pysqlite:///' + environ.get('OCR_FILTER_DB', 'ocr_filter.sqlite'), echo=True)
	metadata.create_all(engine)