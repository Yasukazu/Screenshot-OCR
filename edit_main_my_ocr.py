from typing import Sequence, Optional	
from pickle import loads as pkl_loads

from txt_lines import TTxtLines
from main_my_ocr import Main, calculate_checksum, PathSet
from app_type import AppType
from contextlib import closing
from tool_pyocr import MyOcr
from sys import stderr
from txt_lines_db import sqlite_fullpath
from loguru import logger
logger.remove()
logger.add(stderr, level='INFO', format="{time} | {level} | {message} | {extra}")

class EditMain(Main):
	def __init__(self, app_type=AppType.NUL, db_fullpath=sqlite_fullpath, my_ocr:Optional[MyOcr]=None):
		super().__init__(app_type=app_type, db_fullpath=db_fullpath, my_ocr=my_ocr)

	@classmethod
	def dict_factory(cls, cursor, row):
		d = {}
		for idx, col in enumerate(cursor.description):
			d[col[0]] = row[idx]
		return d

	@logger.catch
	def fix_titles(self, days: Sequence[int], month:int=0, app_type:AppType=AppType.T):
			if not month:
				month = self.my_ocr.date.month
				logger.debug("month==0 is reset as:{}", month)
			with closing(self.conn.cursor()) as cur:
				cur.row_factory = self.dict_factory
				for day in days:
					sql = f"SELECT title, stem, txt_lines, checksum FROM `{self.tbl_name}` WHERE app = {app_type.value} AND day = {day}"
					cur.execute(sql)
					rows = cur.fetchall()
					if len(rows) > 1:
						raise ValueError("Many rows in App and Day!")
					row = rows[0]
					title = row['title']
					yn = input(f"Do you want to keep the title as: '{title}'?:(y/N)")
					if not yn or yn[0].lower() == 'n':
						# load original image
						if not (stem:=row['stem']):
							raise ValueError(f"No stem for the record!")
						img_pathset = PathSet(self.img_dir, stem, '.png')
						img_fullpath = img_pathset.to_path()
						if not img_fullpath.exists():
							raise ValueError(f"`{img_fullpath=}` does not exists!")
						db_chksum = row['checksum']
						if not db_chksum:
							raise ValueError("No checksum in DB row!")
						if calculate_checksum(img_fullpath) != db_chksum:
							raise ValueError("Checksum of the file doesn't match with DB's!")
						txt_lines_pkl = row['txt_lines']
						if not txt_lines_pkl:
							raise ValueError(f"No txt_lines in the row in DB!")
						txt_lines = pkl_loads(txt_lines_pkl)
						if not txt_lines:
							raise ValueError(f"No txt_lines loaded from its pickle!")
						t_txt_lines = TTxtLines(txt_lines=txt_lines, img_pathset=img_pathset, my_ocr=self.my_ocr)
						new_title = t_txt_lines.get_title()
						if not new_title:
							raise ValueError("Unable to get a new title!")
						yn = input(f"Do you want to update to the new title?(Y/n): {new_title}")
						if not yn or yn[0].lower() == 'y':
							# TODO: update DB sequence
							sql = f"UPDATE `{self.tbl_name}` SET title = '{new_title}' WHERE app = {app_type.value} AND day = {day}"
							cur.execute(sql)
							self.conn.commit()
							logger.info("DB is updated app:{}, day:{}, title:{}", app_type, day, new_title)

	@logger.catch
	def fix_titles_from_db(self, days: Sequence[int], month:int=0, app_type:AppType=AppType.T, m_type_title_pos:int=8):
		if not month:
			month = self.my_ocr.date.month
			logger.debug("month==0 is reset as:{}", month)
		with closing(self.conn.cursor()) as cur:
			cur.row_factory = self.dict_factory
			for day in days:
				sql = f"SELECT title, stem, txt_lines, checksum FROM `{self.tbl_name}` WHERE app = {app_type.value} AND day = {day}"
				cur.execute(sql)
				rows = cur.fetchall()
				if len(rows) > 1:
					raise ValueError("Many rows in App and Day!")
				row = rows[0]
				title = row['title']
				yn = input(f"Do you want to keep the title as: '{title}'?:(yes/No)")
				if not yn or yn[0].lower() == 'n':
					txt_lines = row['txt_lines']
					if not txt_lines:
						raise ValueError("No txt_lines in the row in DB!")
					txt_lines = pkl_loads(txt_lines)
					if not txt_lines:
						raise ValueError("No txt_lines loaded from its pickle!")
					new_title = MyOcr.t_title(txt_lines) if app_type == AppType.T else MyOcr.m_title(txt_lines, n=m_type_title_pos)
					if not new_title:
						raise ValueError("Unable to get a new title!")
					yn = input(f"Do you want to update to the new title?(Yes/no): {new_title}")
					if not yn or yn[0].lower() == 'y':
						# TODO: update DB sequence
						sql = f"UPDATE `{self.tbl_name}` SET title = '{new_title}' WHERE app = {app_type.value} AND day = {day}"
						cur.execute(sql)
						self.conn.commit()
						logger.info("DB is updated app:{}, day:{}, title:{}", app_type, day, new_title)

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Fix titles command line options.')
	parser.add_argument('apptype', type=str, help='app_type(T/M)')
	parser.add_argument('-i', '--image', dest='fix_from_image', action='store_const',
						default=False,
						help='fix from image (default: false)')

	args = parser.parse_args(['T']) #, '-i'])
	'''print(args.accumulate(args.integers))
	if len(argv) < 2:
		print("fix_titles: python edit_main_my_ocr.py <app_type>")
		exit(1)'''

	app_to_type = {'T':AppType.T, 'M':AppType.M}
	app_type = app_to_type[args.apptype[0].upper()]
	main = EditMain(app_type=app_type)
	with closing(main.conn.cursor()) as cur:
		sql = f"SELECT `day` FROM `{main.tbl_name}` WHERE LENGTH(`title`) <= 1 AND `app` = {main.app.value}"
		cur.execute(sql)
		days = cur.fetchall()
	days = [int(d[0]) for d in days]
	logger.info("Days to fix: {}", days)
	logger.info("Going to fix from: {}", 'image' if args.fix_from_image else 'DB')
	if args.fix_from_image:
		main.fix_titles(days=days, app_type=app_type)
	else:
		main.fix_titles_from_db(days=days, app_type=app_type)