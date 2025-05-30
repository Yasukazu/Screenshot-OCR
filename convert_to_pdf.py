import re
from contextlib import closing
import sys
from enum import Enum
from pathlib import Path
from typing import Sequence
from pprint import pp

import img2pdf
from dotenv import load_dotenv
load_dotenv()


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

from tool_pyocr import MonthDay
from main_my_ocr import Main
from app_type import AppType
def conv_to_pdf(app: AppType, main: Main):
	main.app = app
	fullpath = main.img_dir / f'202503_{app.name}.pdf'
	sql = f"SELECT `day`, `stem` from `{main.tbl_name}` WHERE `app` = ? ORDER BY `day`;"
	with closing(main.conn.cursor()) as cur:
		result = cur.execute(sql, (app.value,))
	names = [it[1] for it in result]
	convert_to_pdf(output_fullpath=fullpath, stems=names)

from collections import UserList


class MdPage(UserList):
	app = AppType.T
	date_patt = re.compile(r"(\d+)\s*月\s*(\d+)\s*日")
	def __init__(self):#, lst: list[list[str]]):
		super().__init__()
		self._date: MonthDay | None = None

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
				mt = MdPage.date_patt.search(line)
				if mt:
					self._date = MonthDay(int(mt[1]), int(mt[2]))
					return self._date
		raise ValueError('No date block!')

class MMdPage(MdPage):
	"""for Mercari"""
	app = AppType.M
	date_patt = re.compile(r"(\d+)/(\d+)\s*\([月火水木金土日]\)")

	@property
	def title(self):
		for block in self.data:
			if block[0][0] == '【' and block[0][-1] == '】':
				return block[0][1:-1]
		raise ValueError('No M type title found!')
	@property
	def date(self):
		if self._date:
			return self._date
		for block in self.data:
			for line in block:
				mt = MMdPage.date_patt.search(line)
				if mt:
					self._date = MonthDay(int(mt[1]), int(mt[2]))
					return self._date
		raise ValueError('No date block!')

def update_title_in_db_as_md(md: MdPage, main: Main, day: int):
	dt = md.date
	db_date = MonthDay(main.month, day)
	if dt != db_date:
		raise ValueError(f'Mismatch md date:{dt} with db:{db_date}!')
	title = md.title
	if not title:
		raise ValueError("No title in this page!")
	update_title_q = f"UPDATE `{main.tbl_name}` SET `title` = ? WHERE `app` = {md.app.value} and `day` = {dt.day};"
	with closing(main.conn.cursor()) as cur:
		cur.execute(update_title_q, (title,))
	main.conn.commit()


#    options = ["entry 1", "entry 2", "entry 3"]
#    terminal_menu = TerminalMenu(options)
#    menu_entry_index = terminal_menu.show()
def show_title_in_db_then_ask_to_update_with_md(main: Main):#, app=AppType.T):
	app_i = int(input("Input AppType: TM ? -> 1, MH ? -> 2:"))
	app = AppType.T if app_i == 1 else AppType.M
	md_pages = get_pages_from_md(main=main, app=app)
	with closing(main.conn.cursor()) as cur:
		for day, page in md_pages.items():
			get_title_q = f"SELECT `title` FROM `{main.tbl_name}` WHERE `day` = {day} AND `app` = {app.value};"
			ans = cur.execute(get_title_q)
			db_titles = [tt[0] for tt in ans]
			if len(db_titles):
				options = [f"DB title: {db_titles[0]}", f".md title: {page.title}"]
				for i, opt in enumerate(options):
					print(f"{i}. {opt}")
				entry = input(f"Input {[r for r in range(len(options))]} to update DB:")
				if int(entry) > 0:
					sql = f"UPDATE `{main.tbl_name}` SET `title` = ? WHERE `app` = {app.value} and `day` = {day};"
					cur.execute(sql, (page.title,))
					print("A title in MD updated DB.")
			main.conn.commit()


from typing import TextIO
def iter_md_page(reader: TextIO, valid: Sequence[int]=[], pages: dict[int, MdPage] = {}, app: AppType=AppType.T):
	def get_page():
		match app:
			case AppType.T:
				return MdPage()
			case AppType.M:
				return MMdPage()
		raise ValueError('Undefined AppType!')
	page = get_page()
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
					found_date = page.date
					pages[found_date.day] = page
				page.clear()
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
		found_date = page.date
		pages[found_date.day] = page
	return pages

date_patt = re.compile(r"(\d+)月(\d+)日")
def get_day_from_page(page: Sequence[Sequence[str]]):
	for block in page:
		for line in block:
			mt = date_patt.search(line)
			if mt:
				return MonthDay(int(mt[1]), int(mt[2]))

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
	return iter_md_page(reader, app=app)

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
	month = int(sys.argv[1])
	#month = int(input('Month?:'))
	main = Main(month=month)
	from consolemenu import ConsoleMenu
	from consolemenu.items import FunctionItem
	menu = ConsoleMenu("Convert PNG files to a PDF.")
	menu.append_item(FunctionItem('TieMe to PDF', conv_to_pdf, [AppType.T, main]))
	menu.append_item(FunctionItem('MercHal to PDF', conv_to_pdf, [AppType.M, main]))
	menu.append_item(FunctionItem('Fill wages into DB(TM)', fill_wages_in_db, [AppType.T, main]))
	menu.append_item(FunctionItem('Delete invalid rows in DB', delete_null_key_rows, [main]))
	menu.append_item(FunctionItem('Show sum of wages', show_sum_of_wages, [main]))
	menu.append_item(FunctionItem('Update title', show_title_in_db_then_ask_to_update_with_md, [main]))

	menu.show()