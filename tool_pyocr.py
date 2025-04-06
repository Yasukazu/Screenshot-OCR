from contextlib import closing
from enum import IntEnum
from types import MappingProxyType
import re
from typing import Sequence
from collections import namedtuple
from pprint import pp
from pathlib import Path
import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
from PIL import Image, ImageDraw
import pyocr
import pyocr.builders
from pyocr.builders import LineBox
#import cv2
#from google.colab.patches import cv2_imshow
'''[<module 'pyocr.tesseract' from '/home/yasukazu/github/screen/.venv/lib/python3.13/site-packages/pyocr/tesseract.py'>,
 <module 'pyocr.libtesseract' from '/home/yasukazu/github/screen/.venv/lib/python3.13/site-packages/pyocr/libtesseract/__init__.py'>]# '''

@dataclass
class Date:
	month: int
	day: int

def get_date(line_box: pyocr.builders.LineBox):
	content = line_box.content.split()
	if (content[1] != '月') or (content[3] != '日'):
		raise ValueError("Not 月日!")
	return Date(month=int(content[0]), day=int(content[2]))

def next_gyoumu(txt_lines: Sequence[pyocr.builders.LineBox]):
	assert len(txt_lines) > 1
	for n, tx in enumerate(txt_lines):
		joined_tx = ''.join([t.strip() for t in tx.content])
		if joined_tx[0:4] == '業務開始':
			return n
	#return txt_lines[n + 1] #.content


from path_feeder import PathFeeder
from collections import namedtuple
from dataclasses import dataclass
@dataclass
class PathSet:
	path: Path
	stem: str
	ext: str
	def stem_without_delim(self, delim: str):
		return ''.join([s for s in self.stem if s!= delim])

class AppType(IntEnum):
	NUL = 0
	T = 1
	M = 2

APP_TYPE_TO_STEM_END = MappingProxyType({
	AppType.T: ".co.taimee",
	AppType.M: ".mercari.work.android"
})

class MyOcr:
	from path_feeder import input_dir_root
	tools = pyocr.get_available_tools()
	tool = tools[0]
	#input_dir = input_dir # Path(os.environ['SCREEN_BASE_DIR'])
	delim = ' '

	@classmethod
	def get_tool_name(cls):
		return(cls.tool.get_name())	# 'Tesseract (sh)'

	@classmethod
	def get_app_type(cls, txt_lines: Sequence[LineBox]):
		for txt_line in txt_lines:
			if txt_line.content.replace(' ', '').find('おしごと詳細') >= 0:
				return AppType.M
		return AppType.NUL # TODO: check for 'TM'
	@classmethod
	def get_title(cls, app_type: AppType, txt_lines: Sequence[LineBox], n: int):
		match app_type:
			case AppType.T:
				return cls.t_title(txt_lines=txt_lines)
			case AppType.M:
				return cls.m_title(txt_lines, n)
	@classmethod
	def get_wages(cls, app_type: AppType, txt_lines: Sequence[LineBox]):
		match app_type:
			case AppType.T:
				return cls.t_wages(txt_lines=txt_lines)
			case AppType.M:
				for n in range(len(txt_lines)):
					if txt_lines[n].content.replace(' ', '').find('このおしごとの給与'
		) >= 0:
						wages = int(''.join([i for i in txt_lines[n + 1].content.replace(' ', '') if i in '0123456789']))
						if wages in range(1000, 9999):
							return wages
						else:
							raise ValueError("Improper wage!")
			case _: # TODO: check for 'TM'
				raise ValueError("Undefined AppType!")

	M_DATE_PATT_1 = re.compile(r"[^\d]*((\d+)/(\d+))\(")
	M_DATE_PATT_2 = re.compile(r".*時間$")

	@classmethod
	def get_date(cls, app_type: AppType, txt_lines: Sequence[LineBox]):
		match app_type:
			case AppType.M:
				for n in range(len(txt_lines)):
					cntnt = txt_lines[n].content.replace(' ', '')
					mt = MyOcr.M_DATE_PATT_1.match(cntnt)
					mt2 = MyOcr.M_DATE_PATT_2.match(cntnt)
					if mt and mt2:
						grps = mt.groups()
						date = Date(grps[1], grps[2])
						return n, date
				raise ValueError("Unmatch AppType.M txt_lines!")				
			case AppType.T:
				n_gyoumu = next_gyoumu(txt_lines)
				date = get_date(txt_lines[ n_gyoumu])
				return n_gyoumu, date
			case _:
				raise ValueError("Undefined AppType!")

	def __init__(self, month=0):
		self.month = month
		self.path_feeder = PathFeeder(input_dir=MyOcr.input_dir_root, type_dir=False, month=month)
		self.txt_lines: Sequence[pyocr.builders.LineBox] | None = None
		self.image: Image.Image | None = None
	def each_path_set(self):
		for stem in self.path_feeder.feed(delim=self.delim, padding=False):
			yield PathSet(self.path_feeder.dir, stem, self.path_feeder.ext)
	def run_ocr(self, path_set: PathSet, lang='jpn+eng', delim=''):
		#stem_without_delim = ''.join([s for s in path_set[1] if s!= self.delim])
		fullpath = path_set.path / (path_set.stem_without_delim(delim) + path_set.ext)
		self.image = Image.open(fullpath).convert('L')
		self.txt_lines = self.tool.image_to_string(
			self.image,
			lang=lang,
			builder=pyocr.builders.LineBoxBuilder(tesseract_layout=3) # Digit
		)
		return self.txt_lines
	
	def draw_boxes(self):
		if not self.txt_lines:
			raise ValueError('txt_lines is None!')
		if not self.image:
			raise ValueError('Image is None!')
		draw = ImageDraw.Draw(self.image)
		for line in self.txt_lines:
			cood = line.position[0] + line.position[1]
			draw.rectangle(cood, outline=0x55)
		self.image.show()
		return self
	
	def endswith(self, pattern: str):
		for file in self.input_dir_root.iterdir():
			if file.is_file and file.stem.endswith(pattern):
				yield file


	@classmethod
	def t_title(cls, txt_lines: Sequence[LineBox]):
		return ''.join(txt_lines[0].content.split())
	@classmethod
	def m_title(cls, txt_lines: Sequence[LineBox], n: int):
		return ''.join([txt_lines[i].content.replace(' ', '') for i in range(n - 3, n - 1)])

	'''def get_date(self, app_tpe: AppType):
		if not self.txt_lines:
			raise bb ValueError('`txt_lines` is None!')
		match
		n_gyoumu = next_gyoumu(self.txt_lines)
		return get_date(n_gyoumu)'''
	@classmethod
	def t_wages(cls, txt_lines: Sequence[LineBox]):
		content = txt_lines[-1].content
		content_num = ''.join(re.findall(r'\d+', content)) # ''.join([n for n in content if '0123456789'.index(n) >= 0])
		try:
			num = int(content_num)
			if not (1000 < num < 9999):
				raise ValueError(f"Unexpected value: {num}")
			return num
		except ValueError as err:
			raise ValueError(f"Failed to convert into an integer: {content_num}\n{err}")

	def check_date(self, path_set: PathSet):#, txt_lines: Sequence[pyocr.builders.LineBox]):
		if not self.txt_lines:
			raise ValueError('txt_lines is None!')
		n_gyoumu = next_gyoumu(self.txt_lines)
		gyoumu_date = get_date(self.txt_lines[n_gyoumu])
		if gyoumu_date != (int(path_set.stem.split()[0]), int(path_set.stem.split()[1])): # path_feeder.month, 
			raise ValueError(f"Unmatch {gyoumu_date} : {path_set.stem}!")

'''for w_box in n_gyoumu.word_boxes:
	pp(w_box.content)
for tx in txt_lines:
	print(tx)'''

import pickle
import sqlite3


class Main:
	def __init__(self, month=0, app=AppType.NUL):
		self.my_ocr = MyOcr(month=month)
		self.img_dir = self.my_ocr.path_feeder.dir
		month = self.my_ocr.month
		img_parent_dir = self.img_dir.parent
		self.app = app # tm
		txt_lines_db = TxtLinesDB(img_parent_dir=img_parent_dir)
		self.conn = txt_lines_db.conn
		create_tbl_sql = f"CREATE TABLE if not exists `{self.tbl_name}` (app INTEGER, day INTEGER, wages INTEGER, title TEXT, stem TEXT, txt_lines BLOB, PRIMARY KEY (app, day))"
		with closing(self.conn.cursor()) as cur:
			cur.execute(create_tbl_sql)
		self.conn.commit()

	@property
	def month(self):
		return self.my_ocr.month
	@property
	def tbl_name(self):
		return Main.get_table_name(self.month)
	def sum_wages(self):
		with closing(self.conn.cursor()) as cur:
			sum_q = f"SELECT SUM(wages) from `{self.tbl_name}`"
			result = cur.execute(sum_q)
			if result:
				return [r[0] for r in result][0]

	def get_existing_days(self, app_type: AppType):
		qry = f"SELECT day from `{self.tbl_name}` where app = ?;"
		prm = (app_type.value, ) # date.day)
		with closing(self.conn.cursor()) as cur:
			result = cur.execute(qry, prm)
			if result:
				return [r[0]	for r in result]

	def add_image_files_as_txt_lines(self, app_type: AppType, ext='.png'):
		if app_type == AppType.NUL:
			raise ValueError('AppType is NUL!')
		ocr_done = []
		for img_file in self.img_dir.glob('*' + ext):
			if not img_file.stem.endswith(APP_TYPE_TO_STEM_END[app_type]):
				continue
			#ext_dot = img_file.name.rfind('.')
			#ext = img_file.name[ext_dot:]
			stem = img_file.stem
			parent = self.my_ocr.path_feeder.dir
			path_set = PathSet(parent, stem, ext)
			txt_lines = self.my_ocr.run_ocr(path_set=path_set, delim='')
			if not txt_lines:
				raise ValueError(f"Unable to extract from {path_set}")
			# app_type = self.my_ocr.get_app_type(txt_lines) if txt_lines else AppType.NUL
			n, date = self.my_ocr.get_date(app_type=app_type, txt_lines=txt_lines)
			existing_day_list = self.get_existing_days(app_type=app_type)
			if existing_day_list and (date.day in existing_day_list):
				print(f"Day {date.day} of App {app_type} exists.")
			else:
				wages = self.my_ocr.get_wages(app_type=app_type, txt_lines=txt_lines)
				title = self.my_ocr.get_title(app_type, txt_lines, n)
				pkl = pickle.dumps(txt_lines)
				with closing(self.conn.cursor()) as cur:
					cur.execute(f"INSERT INTO `{self.tbl_name}` VALUES (?, ?, ?, ?, ?, ?);", (app_type.value, date.day, wages, title, stem, pkl))
				for line in txt_lines:
					pp(line.content)
				self.conn.commit()
				ocr_done.append((app_type, date))
		return ocr_done
	def add_image_file_without_content_into_db(self, app_type: AppType, stem: str, date: Date, wages=None, title=None, pkl=None):
		with closing(self.conn.cursor()) as cur:
			cur.execute(f"INSERT INTO `{self.tbl_name}` VALUES (?, ?, ?, ?, ?, ?);", (app_type.value, date.day, wages, title, stem, pkl))
		self.conn.commit()


if __name__ == '__main__':
	import sys
	app = int(sys.argv[1])
	month = int(sys.argv[2])
	main = Main(month=month, app=app)
	if len(sys.argv) > 4:
		day = int(sys.argv[3])
		stem = sys.argv[4]
		match app:
			case 1:
				app_type = AppType.T
			case 2:
				app_type = AppType.M
			case _:
				raise ValueError('Unsupported app type!')
		main.add_image_file_without_content_into_db(app_type, stem, Date(month=month, day=day))
		sys.exit(0)
	from consolemenu import ConsoleMenu, SelectionMenu
	from consolemenu.items import FunctionItem, SubmenuItem
	import consolemenu.items
	def show_total_wages():
		print(main.sum_wages())
		input('Hit Enter key, please:')
	def show_existing_days():
		app = input("App type: Tm: 1, MH: 2")
		app_type = [AppType.NUL, AppType.T, AppType.M][int(app)]
		print(main.get_existing_days(app_type ))
		input('Hit Enter key, please:')
	def run_ocr():
		app_type_list = [AppType.T, AppType.M]
		patt_list = [APP_TYPE_TO_STEM_END[AppType.T], APP_TYPE_TO_STEM_END[AppType.M]]#["t.*.png", "Screenshot_*.png"]
		selected = SelectionMenu.get_selection([APP_TYPE_TO_STEM_END[at] for at in app_type_list])
		txt_lines = main.add_image_files_as_txt_lines(
			app_type_list[selected]
		)
		txt_lines_len = len(txt_lines) if txt_lines else 0
		input(f'{txt_lines_len} file(s) is / are OCRed. Hit Enter key to return to the main menu:')
	function_items = [
		FunctionItem("Show the total wages", show_total_wages),
		FunctionItem("Show existing days", show_existing_days),# ["Enter"]),
	]
	menu = ConsoleMenu("Menu", "-- OCR DB --")
	for f_it in function_items:
		menu.append_item(f_it)
	submenu = ConsoleMenu("SubMenu", "-- Run OCR --")
	submenu.append_item(FunctionItem("Run OCR with file name patterns:", run_ocr))
	submenu_item = SubmenuItem("SubMenu", submenu=submenu, menu=menu)
	menu.append_item(submenu_item)
	
	menu.show()
