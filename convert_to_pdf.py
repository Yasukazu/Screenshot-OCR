from enum import Enum
from pathlib import Path
from typing import Sequence, Iterator, Generator

from PIL import Image, ImageDraw
import img2pdf


from path_feeder import FileExt, PathFeeder, get_last_month #, YearMonth


class PdfLayout(Enum):
	a4pt = (img2pdf.mm_to_pt(210),img2pdf.mm_to_pt(297))
	a3lp = (img2pdf.mm_to_pt(420),img2pdf.mm_to_pt(297))


def convert_to_pdf(output_fullpath: Path, stems: Sequence[str], layout=PdfLayout.a4pt):
	layout_fun = img2pdf.get_layout_fun(layout.value)
	parent = output_fullpath.parent
	names = [str(parent / stem)+'.png' for stem in stems]
	with output_fullpath.open("wb") as f:
		buff = img2pdf.convert(layout_fun=layout_fun, *names, rotation=img2pdf.Rotation.ifvalid)
		if buff:
			f.write(buff)

if __name__ == '__main__':
	from dotenv import load_dotenv
	load_dotenv()
	import os, sys
	block = int(sys.argv[1])
	input_dir = Path(os.environ['SCREEN_BASE_DIR']) / '2025' / '03' 
	fullpath = input_dir/ '202503_T_images-1.pdf'
	from tool_pyocr import Main
	main = Main(month=3, app=1)
	cur = main.con.cursor()
	result = cur.execute(f"SELECT `day`, `stem` from `{main.tbl_name}` ORDER BY `day`;")
	frm = to = 0
	if block == 2:
		frm = 10
		to = 21
	names = [it[1] for it in result][frm: to] # if it[0] <= 10]
	convert_to_pdf(output_fullpath=fullpath, stems=names)