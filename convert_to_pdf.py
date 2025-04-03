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
	with closing(main.con.cursor()) as cur:
		result = cur.execute(sql, (app.value,))
	names = [it[1] for it in result]
	convert_to_pdf(output_fullpath=fullpath, stems=names)

from typing import TextIO
def iter_md_page(reader: TextIO, valid: Sequence[int]):
	pages = {}
	page = []
	block = []
	pg = 1
	lines = reader.readlines()
	for line in lines:
		ln = line.strip() 
		if ln == '<br />':#'\u2120':
			if len(block) > 0:
				page.append(block)
			if pg in valid:
				pages[pg] = page
			page = []
			block = []
			pg += 1
			continue
		if len(ln) > 0:
			block.append(ln)
		else:
			if len(block) > 0:
				page.append(block)
				block = []
	if len(page) > 0 and pg in valid:
		if len(block) > 0:
			page.append(block)
		pages[pg] = page
	return pages

import re
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

def fill_wages_in_db(app: AppType, main: Main, day=0):
	lacking_wages_q = f"SELECT `day` from `{main.tbl_name}` WHERE `app` = ? AND `wages` IS NULL ORDER BY `day`;"
	days = [day]
	main.app = app
	if not day:
		with closing(main.con.cursor()) as cur:
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
		with closing(main.con.cursor()) as cur:
			inserted = cur.execute(put_wages_q, (wages, title))
			print(f"Inserted wages:{[i for i in inserted]}")
		main.con.commit()

def delete_null_key_rows(main: Main):
	delete_qry = f"DELETE FROM `{main.tbl_name}` WHERE `app` IS NULL OR `day` IS NULL;"
	with closing(main.con.cursor()) as cur:
		r = cur.execute(delete_qry)
		pp(r)
	main.con.commit()

def show_sum_of_wages(main: Main):
	q = f"SELECT SUM(`wages`) FROM `{main.tbl_name}`;"
	with closing(main.con.cursor()) as cur:
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
	menu.show()