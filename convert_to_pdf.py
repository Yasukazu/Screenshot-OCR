import re
from contextlib import closing
import os, sys
from enum import Enum
from pathlib import Path
from typing import Sequence, Iterator, Generator
from pprint import pp

from PIL import Image, ImageDraw
import img2pdf
from dotenv import load_dotenv
load_dotenv()

from path_feeder import FileExt, PathFeeder, get_last_month #, YearMonth

class PdfLayout(Enum):
	a4pt = (img2pdf.mm_to_pt(210),img2pdf.mm_to_pt(297))
	a3lp = (img2pdf.mm_to_pt(420),img2pdf.mm_to_pt(297))

def convert_to_pdf(output_fullpath: Path, stems: Sequence[str], layout=PdfLayout.a4pt, separator=False):
	layout_fun = img2pdf.get_layout_fun(layout.value)
	parent = output_fullpath.parent
	names = [str(parent / stem)+'.png' for stem in stems]
	with output_fullpath.open("wb") as f:
		buff = img2pdf.convert(layout_fun=layout_fun, *names, rotation=img2pdf.Rotation.ifvalid)
		if buff:
			f.write(buff)

from tool_pyocr import AppType, Main, Date
from path_feeder import input_dir

def conv_to_pdf(app: AppType, main: Main):
	main.app = app
	# input_dir = Path(os.environ['SCREEN_BASE_DIR']) / '2025' / '03' 
	# app_c = 'T' if app == 1 else 'M'
	fullpath = main.img_dir / f'202503_{app.name}.pdf'
	sql = f"SELECT `day`, `stem` from `{main.tbl_name}` WHERE `app` = ? ORDER BY `day`;"
	with closing(main.conn.cursor()) as cur:
		result = cur.execute(sql, (app.value,))
	names = [it[1] for it in result]
	convert_to_pdf(output_fullpath=fullpath, stems=names)

from collections import UserList

date_patt = re.compile(r"(\d+)月(\d+)日")

class MdPage(UserList):
	app = AppType.T
	def __init__(self):
		super().__init__()
		self._date: Date | None = None

	def get_block(self, block: int):
		return self.data[block]
	
	@property
	def title(self):
		return '-'.join(self.get_block(0))

	@property
	def date(self):
		if self._date:
			return self._date
		for block in self.data:
			for line in block:
				mt = date_patt.search(line)
				if mt:
					self._date = Date(int(mt[1]), int(mt[2]))
					return self._date
		raise ValueError('No date block!')
		
def update_title_in_db_as_md(md: MdPage, main: Main, day: int):
	dt = md.date
	db_date = Date(main.month, day)
	if dt != db_date:
		raise ValueError(f'Mismatch md date:{dt} with db:{db_date}!')
	title = md.title
	if not title:
		raise ValueError("No title in this page!")
	update_title_q = f"UPDATE `{main.tbl_name}` SET `title` = ? WHERE `app` = {md.app.value} and `day` = {dt.day};"
	with closing(main.conn.cursor()) as cur:
		cur.execute(update_title_q, (title,))
	main.conn.commit()

def show_title_in_db_then_ask_to_update_with_md(main: Main, app=AppType.T):
	pages = get_pages_from_md(main=main, app=app)
	with closing(main.conn.cursor()) as cur:
		for day, page in pages.items():
			get_title_q = f"SELECT `title` FROM `{main.tbl_name}` WHERE `day` = {day} AND `app` = {app.value};"
			ans = cur.execute(get_title_q)
			titles = [tt[0] for tt in ans]
			if len(titles) > 1:
				raise ValueError("Many titles for (day:{day}, app:{app.name})!")
			print(f"DB title: {titles[0]}")
			print(f".md title: {page.title}")
			a = input(f"Like to update as md one?(Y/N):")
			try:
				if a.upper() == 'Y':
					print("Starting DB update..")
			except ValueError:
				print("Invalid answer.")


from typing import TextIO
def iter_md_page(reader: TextIO, valid: Sequence[int]=[], pages: dict[int, MdPage] = {}):
	page = MdPage()
	block: list[str] = []
	pg = 1
	def is_valid(pg):
		if not valid:
			return True
		if pg in valid:
			return True
		return False
	lines = reader.readlines()
	for line in lines:
		ln = line.strip()
		if len(ln) > 0:
			if ln == '<br />':#'\u2120':
				if len(block) > 0:
					page.append(block)
				if is_valid(pg):
					pages[pg] = page
				page = MdPage()
				block = []
				pg += 1
			else:
				block.append(ln)
		else:
			if len(block) > 0:
				page.append(block)
				block = []
	if len(page) > 0 and is_valid(pg):
		if len(block) > 0:
			page.append(block)
		pages[pg] = page
	return pages

date_patt = re.compile(r"(\d+)月(\d+)日")
def get_day_from_page(page: Sequence[Sequence[str]]):
	for block in page:
		for line in block:
			mt = date_patt.search(line)
			if mt:
				return Date(int(mt[1]), int(mt[2]))

def get_wages_from_page(page: Sequence[Sequence[str]]):
	for block in page:
		for line in block:
			if line == '差引支給額':
				num_line = block[1]
				nums = [n for n in num_line if n in '0123456789']
				return int(''.join(nums))

def get_pages_from_md(main: Main, app: AppType):
	fullpath = main.img_dir / f'202503_{app.name}.md'
	reader = fullpath.open()
	return iter_md_page(reader)

def fill_wages_in_db(app: AppType, main: Main, day=0):
	lacking_wages_q = f"SELECT `day` from `{main.tbl_name}` WHERE `app` = ? AND `wages` IS NULL ORDER BY `day`;"
	days = [day]
	main.app = app
	if not day:
		with closing(main.conn.cursor()) as cur:
			result = cur.execute(lacking_wages_q, (app.value,))
			if not result:
				print("No lacking wages found.")
				return
			days = [r[0] for r in result]
	fullpath = main.img_dir / f'202503_{app.name}.md'
	reader = fullpath.open()
	pages = iter_md_page(reader, days)
	for pg, page in pages.items():
		dt = get_day_from_page(page)
		if not dt:
			raise ValueError(f"No date is found in page {pg}")
		if dt.day != pg:
			raise ValueError(f"Mismatch date {dt.day} : page {pg}")
		wages = get_wages_from_page(page)
		title = page[0][0]
		if not title:
			raise ValueError("No title in the page!")
		put_wages_q = f"UPDATE `{main.tbl_name}` SET `wages` = ?, `title` = ? WHERE `app` = {app.value} and `day` = {dt.day};"
		with closing(main.conn.cursor()) as cur:
			inserted = cur.execute(put_wages_q, (wages, title))
			print(f"Inserted wages:{[i for i in inserted]}")
		main.conn.commit()

def delete_null_key_rows(main: Main):
	delete_qry = f"DELETE FROM `{main.tbl_name}` WHERE `app` IS NULL OR `day` IS NULL;"
	with closing(main.conn.cursor()) as cur:
		r = cur.execute(delete_qry)
		pp(r)
	main.conn.commit()

def show_sum_of_wages(main: Main):
	q = f"SELECT SUM(`wages`) FROM `{main.tbl_name}`;"
	with closing(main.conn.cursor()) as cur:
		rr = cur.execute(q)
		if not rr:
			raise ValueError("No wages!")
		print([r[0] for r in rr])
		input('Hit Enter key:')


if __name__ == '__main__':
	month = int(sys.argv[1]) #block
	day = int(input('Day?:'))
	main = Main(month=month)
	from consolemenu import ConsoleMenu
	from consolemenu.items import FunctionItem
	menu = ConsoleMenu("Convert PNG files to a PDF.")
	menu.append_item(FunctionItem('TieMe to PDF', conv_to_pdf, [AppType.T, main]))
	menu.append_item(FunctionItem('MercHal to PDF', conv_to_pdf, [AppType.M, main]))
	menu.append_item(FunctionItem('Fill wages into DB(TM)', fill_wages_in_db, [AppType.T, main, day]))
	menu.append_item(FunctionItem('Delete invalid rows in DB', delete_null_key_rows, [main]))
	menu.append_item(FunctionItem('Show sum of wages', show_sum_of_wages, [main]))
	menu.append_item(FunctionItem('Update title', show_title_in_db_then_ask_to_update_with_md, [main]))

	menu.show()