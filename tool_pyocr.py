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
class MyOcr:
	tools = pyocr.get_available_tools()
	tool = tools[0]
	input_dir = Path(os.environ['SCREEN_BASE_DIR'])
	delim = ' '

	@classmethod
	def get_tool_name(cls):
		return(cls.tool.get_name())	# 'Tesseract (sh)'

	def __init__(self, month=0):
		self.path_feeder = PathFeeder(input_dir=MyOcr.input_dir, type_dir=False, month=month)
		self.txt_lines: Sequence[pyocr.builders.LineBox] | None = None
		self.image: Image.Image | None = None
	def each_path_set(self):
		for stem in self.path_feeder.feed(delim=self.delim, padding=False):
			yield PathSet(self.path_feeder.dir, stem, self.path_feeder.ext)
	def run_ocr(self, path_set: PathSet, lang='jpn+eng', delim=' '):
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

def main():
	my_ocr = MyOcr()
	for path_set in my_ocr.each_path_set():
		my_ocr.run_ocr(path_set=path_set).draw_boxes().check_date(path_set)
		pp(path_set, my_ocr.date)


if __name__ == '__main__':
	import sys
	import shelve
	import pickle
	SHELVE_HDR = 'shlv-'
	month = 3
	my_ocr = MyOcr(month=month)
	img_dir = my_ocr.path_feeder.dir
	img_parent_dir = img_dir.parent
	sqlite_name = 'txt_lines.sqlite'
	sqlite_fullpath = img_parent_dir / sqlite_name
	shelv_name = f"{SHELVE_HDR}{month:02}"
	shelv_fullpath = img_dir / shelv_name
	import sqlite3
	app = 1 # tm
	con = sqlite3.connect(str(sqlite_fullpath))
	cur = con.cursor()
	tbl_name = f"text_lines-{month:02}"
	create_tbl_sql = f"CREATE TABLE if not exists `{tbl_name}` (app INTEGER, day INTEGER, wages INTEGER, title TEXT, stem TEXT, txt_lines BLOB, PRIMARY KEY (app, day))"
	cur.execute(create_tbl_sql)
	# with shelve.open(shelv_fullpath) as shlv:
	for img_file in img_dir.glob("*.png"):
		ext_dot = img_file.name.rfind('.')
		stem = img_file.stem
		ext = img_file.name[ext_dot:]
		parent = my_ocr.path_feeder.dir
		path_set = PathSet(parent, stem, ext) # sys.argv[1], '.jpg')
		txt_lines = my_ocr.run_ocr(path_set=path_set, delim='')
		date = my_ocr.date
		date_str = f"{date[0]:02}{date[1]:02}"
		pkl = pickle.dumps(txt_lines)
		cur.execute(f"INSERT INTO `{tbl_name}` VALUES (?, ?, ?, ?, ?, ?);", (app, date.day, my_ocr.wages, my_ocr.title, stem, pkl))
		# shlv[date_str] = txt_lines
		for line in txt_lines:
			pp(line.content)
		con.commit()