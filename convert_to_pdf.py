from enum import Enum
from pathlib import Path
from typing import Sequence, Iterator, Generator

from PIL import Image, ImageDraw
import img2pdf


from path_feeder import FileExt, PathFeeder, get_last_month #, YearMonth


class PdfLayout(Enum):
	a4pt = (img2pdf.mm_to_pt(210),img2pdf.mm_to_pt(297))
	a3lp = (img2pdf.mm_to_pt(420),img2pdf.mm_to_pt(297))


def convert_to_pdf(fullpath: Path, names: Sequence[str], layout=PdfLayout.a4pt):
	layout_fun = img2pdf.get_layout_fun(layout.value)
	with fullpath.open("wb") as f:
		buff = img2pdf.convert(layout_fun=layout_fun, *names, rotation=img2pdf.Rotation.ifvalid)
		if buff:
			f.write(buff)

if __name__ == '__main__':
	from dotenv import load_dotenv
	load_dotenv()
	import os
	input_dir = Path(os.environ['SCREEN_BASE_DIR']) / '2025' / '03' 
	fullpath = input_dir/ 't._202503_png.pdf'
	names = [str(n) for n in input_dir.glob("t._202503*.png")]
	convert_to_pdf(fullpath=fullpath, names=names)