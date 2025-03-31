from enum import IntEnum
import re
from typing import Sequence
from collections import namedtuple
from pprint import pp
from pathlib import Path
import os
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

Date = namedtuple('Date', ['month', 'day'])

def get_date(line_box: pyocr.builders.LineBox):
	content = line_box.content.split()
	if (content[1] != '月') or (content[3] != '日'):
		raise ValueError("Not 月日!")
	return Date(month=int(content[0]), day=int(content[2]))

def next_gyoumu(txt_lines: Sequence[pyocr.builders.LineBox]):
	for n, tx in enumerate(txt_lines):
		joined_tx = ''.join([t.strip() for t in tx.content])
		if joined_tx[0:4] == '業務開始':
			break
	return txt_lines[n + 1] #.content


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
class MyOcr:
	tools = pyocr.get_available_tools()
	tool = tools[0]
	input_dir = Path(os.environ['SCREEN_BASE_DIR'])
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
	def get_wages(cls, app_type: AppType, txt_lines: Sequence[LineBox]):
		match app_type:
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
	M_DATE_PATT_1 = re.compile(r"(\d+)/(\d+)\(")
	M_DATE_PATT_2 = re.compile(r"時間$")
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
						date = Date(grps[0], grps[1])
						return date
						#else:							raise ValueError("Improper wage!")
			case _: # TODO: check for 'TM'
				raise ValueError("Undefined AppType!")

	def __init__(self, month=0):
		self.month = month
		self.path_feeder = PathFeeder(input_dir=MyOcr.input_dir, type_dir=False, month=month)
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

	@property
	def title(self):
		if not self.txt_lines:
			raise ValueError('`txt_lines` is None!')
		return ''.join(self.txt_lines[0].content.split())

	@property
	def date(self):
		if not self.txt_lines:
			raise ValueError('`txt_lines` is None!')
		n_gyoumu = next_gyoumu(self.txt_lines)
		return get_date(n_gyoumu)
	@property
	def wages(self):
		if not self.txt_lines:
			raise ValueError('`txt_lines` is None!')
		content = self.txt_lines[-1].content

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
		gyoumu_date = get_date(n_gyoumu)
		if gyoumu_date != (int(path_set.stem.split()[0]), int(path_set.stem.split()[1])): # path_feeder.month, 
			raise ValueError(f"Unmatch {gyoumu_date} : {path_set.stem}!")

'''for w_box in n_gyoumu.word_boxes:
	pp(w_box.content)
for tx in txt_lines:
	print(tx)'''

import pickle
import sqlite3

class Main:
	sqlite_name = 'txt_lines.sqlite'
	def __init__(self, month=0, app=1):
		self.month = month
		self.my_ocr = MyOcr(month=month)
		self.img_dir = self.my_ocr.path_feeder.dir
		img_parent_dir = self.img_dir.parent
		sqlite_fullpath = img_parent_dir / Main.sqlite_name
		self.app = app # tm
		if sqlite3.threadsafety == 3:
			check_same_thread = False
		else:
			check_same_thread = True
		self.con = sqlite3.connect(str(sqlite_fullpath), check_same_thread=check_same_thread)
		self.tbl_name = f"text_lines-{self.month:02}"
		create_tbl_sql = f"CREATE TABLE if not exists `{self.tbl_name}` (app INTEGER, day INTEGER, wages INTEGER, title TEXT, stem TEXT, txt_lines BLOB, PRIMARY KEY (app, day))"
		cur = self.con.cursor()
		cur.execute(create_tbl_sql)

	def sum_wages(self):
		cur = self.con.cursor()
		sum_q = f"SELECT SUM(wages) from `{self.tbl_name}`"
		result = cur.execute(sum_q)
		if result:
			return [r[0] for r in result][0]

	def get_existing_days(self):
		qry = f"SELECT day from `{self.tbl_name}` where app = ?;"
		prm = (self.app, ) # date.day)
		cur = self.con.cursor()
		result = cur.execute(qry, prm)
		if result:
			return [r[0]	for r in result]

	def add_image_files_as_txt_lines(self, pattern='*.png'):
		ocr_done = 0
		for img_file in self.img_dir.glob(pattern):
			ext_dot = img_file.name.rfind('.')
			stem = img_file.stem
			ext = img_file.name[ext_dot:]
			parent = self.my_ocr.path_feeder.dir
			path_set = PathSet(parent, stem, ext)
			txt_lines = self.my_ocr.run_ocr(path_set=path_set, delim='')
			app = self.my_ocr.get_app_type(txt_lines) if txt_lines else 0
			date = self.my_ocr.date
			# check if exist
			existing_day_list = self.get_existing_days()
			if not existing_day_list:
				return
			if date.day in existing_day_list:
				print(f"Day {date.day} exists.")
			else:
				txt_lines = self.my_ocr.run_ocr(path_set=path_set)
				if not txt_lines:
					raise ValueError(f"Unable to extract from {path_set}")
				pkl = pickle.dumps(txt_lines)
				cur = self.con.cursor()
				cur.execute(f"INSERT INTO `{self.tbl_name}` VALUES (?, ?, ?, ?, ?, ?);", (app, date.day, self.my_ocr.wages, self.my_ocr.title, stem, pkl))
				for line in txt_lines:
					pp(line.content)
				self.con.commit()
				ocr_done += 1
		return ocr_done

if __name__ == '__main__':
	import sys
	month = int(sys.argv[1])
	app = int(sys.argv[2])
	main = Main(month=month, app=app)
	from consolemenu import ConsoleMenu, SelectionMenu
	from consolemenu.items import FunctionItem, SubmenuItem
	import consolemenu.items
	def show_total_wages():
		print(main.sum_wages())
		input('Hit Enter key, please:')
	def show_existing_days():
		print(main.get_existing_days())
		input('Hit Enter key, please:')
	def run_ocr():
		patt_list = ["t.*.png", "Screenshot_*.png"]
		selected = SelectionMenu.get_selection(patt_list)
		txt_lines = main.add_image_files_as_txt_lines(pattern=patt_list[selected])
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
