from enum import Enum
from typing import Sequence, Iterator, Generator
import numpy as np
from PIL import Image, ImageDraw
import img2pdf
import os.path, calendar
import ttl
from pathlib import Path
home_dir = Path(os.path.expanduser('~'))
import path_feeder
from path_feeder import PathFeeder, get_last_month #, YearMonth
from digit_image import ImageFill
from put_number import PutPos, put_number
last_month_date = get_last_month()
year = last_month_date.year
month = last_month_date.month
img_dir = home_dir / 'Documents' / 'screen' / str(year) / f'{month:02}'
if not img_dir.exists():
	img_dir.mkdir()

IMG_SIZE = (720, 1612)
H_PAD = 20
V_PAD = 40

file_over = False

year_month_name = f"{year}-{month:02}"

class PdfLayout(Enum):
	a4pt = (img2pdf.mm_to_pt(210),img2pdf.mm_to_pt(297))
	a3lp = (img2pdf.mm_to_pt(420),img2pdf.mm_to_pt(297))

def convert_tiff_to_png_files():
	cmd = 'convert 2025-02-8x4.tif -format png -scene 1 2025-02-%d.png'
from path_feeder import FileExt
def paged_png_feeder(layout=PdfLayout.a4pt):
	feeder=PathFeeder(input_type=FileExt.QPNG)
	img_dir = feeder.dir
	year = feeder.year
	month = feeder.month
	names = []
	match layout:
		case PdfLayout.a4pt:
			for p in range(4):
				fullpath = img_dir / f"{year}-{month:02}-{p + 1}.png"
				if not fullpath.exists():
					ValueError(f"{fullpath} does not exist!")
				names.append(fullpath)
			return names
		case PdfLayout.a3lp:
			for p in range(2):
				fullpath_l = img_dir / f"{year}-{month:02}-{2 * p + 1}.png"
				fullpath_r = img_dir / f"{year}-{month:02}-{2 * p + 2}.png"
				for fullpath in (fullpath_l, fullpath_r):
					if not fullpath.exists():
						ValueError(f"{fullpath} does not exist!")
				sub_ext_list = ('L', 'R')
				outpath = img_dir / f"{year}-{month:02}-{sub_ext_list[p]}.png"
				l_image = Image.open(fullpath_l)
				padding = 100
				margin = 200
				image = Image.new('L', (l_image.width * 2 + padding + 2 * margin, l_image.height))
				margin_img = Image.new('L', (margin, l_image.height), (0xff,))
				pos = 0
				image.paste(margin_img, (0, 0))
				pos += margin_img.width
				image.paste(l_image, (pos, 0))
				pos += l_image.width + padding
				l_image = Image.open(fullpath_r)
				image.paste(l_image, (pos, 0))
				pos += l_image.width
				image.paste(margin_img, (pos, 0))
				save_path = (outpath)
				image.save(save_path)
				names.append(save_path)
			return names

def convert_to_pdf(layout=PdfLayout.a4pt):
	names = paged_png_feeder(layout=layout)
	parent_dir = names[0].parent
	fullpath = parent_dir / f"{year}-{month:02}.pdf"
	layout_fun = img2pdf.get_layout_fun(layout.value)
	with open(fullpath,"wb") as f:
		for name in names:
			assert name.exists()
		name_list = [str(n) for n in names]
		f.write(img2pdf.convert(layout_fun=layout_fun, *name_list, rotation=img2pdf.Rotation.ifvalid))

from path_feeder import PathFeeder
def save_pages_as_pdf(): #fullpath=PathFeeder().first_fullpath):
	fullpath = img_dir / f"{year_month_name}.pdf"
	imges = list(draw_onto_pages())
	imges[0].save(fullpath, "PDF" ,resolution=200, save_all=True, append_imges=imges[1:])

from path_feeder import ext_to_dir, FileExt, ExtDir
def save_qpng_pages(ext_dir=FileExt.QPNG):
	save_dir = img_dir / ext_dir.value.dir
	for pg, img in enumerate(draw_onto_pages()):
		fullpath = save_dir / f"8-img-{pg + 1}{ext_dir.value.ext}"
		img.save(fullpath) #, 'PNG')

class ArcFileExt(Enum):
	TIFF = ('.tif', {'compression':"tiff_deflate"})
	PDF = ('.pdf', {})

def save_arc_pages(ext: ArcFileExt=ArcFileExt.TIFF):
		imgs: list[Image.Image] = list(draw_onto_pages())
		fullpath = img_dir / f"{year}-{month:02}-8x4{ext.value[0]}"
		imgs[0].save(fullpath, save_all=True, append_images=imgs[1:], **ext.value[1])

from typing import Generator
from path_feeder import path_feeder, FileExt
def get_png_file_names()-> Generator[tuple[Path, str, int], None, None]:
	for path, stem, m in path_feeder(input_type=FileExt.PNG):
		yield path, stem, m

def get_quad_png_file_names()-> Generator[tuple[Path, str, int], None, None]:
	for path, stem, m in path_feeder(input_type=FileExt.QPNG):
		yield path, stem, m

DAY_NOMBRE_H = 50
TXT_OFST = 0 # width-direction / horizontal

from digit_image import BASIC_DIGIT_IMAGE_PARAM_LIST, BasicDigitImage

def draw_onto_pages(div=64, th=H_PAD // 2,
	path_feeder: PathFeeder=PathFeeder(),
	v_pad=16, h_pad=8, mode='L', dst_bg=ImageFill.BLACK)-> Iterator[Image.Image]:
	# name_feeder = path_feeder.feed
	first_fullpath = path_feeder.first_fullpath
	if not first_fullpath:
		raise ValueError(f"No '{path_feeder.ext}' file in {path_feeder.dir}!")
	first_img_size = Image.open(first_fullpath).size

	drc_tbl = [(0, -1), (1, 0), (0, 1), (-1, 0)]
	ll_ww = [0, 0]
	def _to(n, xy):
		ll, ww = ll_ww[0], ll_ww[1]
		x = xy[0]
		y = xy[1]
		drc = drc_tbl[n]
		return x + ll * drc[0], y + ll * drc[1]
	def xshift(offset, *xy):
		return xy[0] + offset, xy[1]
	def yshift(*xy):
		x, y = xy[0], xy[1]
		return x, y - (V_PAD - 8)
	# fill_white = (255,255,255)
	def page_dots(ct, drw, n):
		ll, ww = ll_ww[0], ll_ww[1]
		op = list(ct)
		for i in range(n):
			dp = op[0] - ww, op[1]
			drw.line((*op, *dp), fill=ImageFill.invert(dst_bg).value, width=int(th))
			op[0] -= 2 * ww
	def month_dots(ct, drw, img_w):
		ttl.set_pit_len(img_w // 32)
		ttl.set_org(*(ct[0] + (img_w // 32), ct[1]))
		for frm, to in ttl.plot(month, ttl.Direc.RT):
			drw.line((frm[0], frm[1], to[0], to[1]), fill=ImageFill.invert(dst_bg).value, width=int(th))

	def get_image_blocks():
		names = list(path_feeder.feed(padding=True))
		name_blocks: Sequence[list[str]] = [names[:8], names[8:16], names[16:24], names[24:]]
		pad_size = 8 - len(name_blocks[-1])
		name_blocks[-1] += [''] * pad_size
		for i, block in enumerate(name_blocks):
			yield concat_8_pages(block, number_str=f"{path_feeder.month:02}{(-0xa - i):x}")

	digit_image_feeder_L = BasicDigitImage(scale=36, line_width=8, padding=(6, 6), bgcolor=ImageFill.BLACK)
	# digit_image_param_L = BASIC_DIGIT_IMAGE_PARAM_LIST[1]
	from put_number import put_number
	
	@put_number(pos=PutPos.R, digit_image_feeder=digit_image_feeder_L)
	def concat_8_pages(names: list[str], number_str: str)-> Image.Image:

		names_1 = list(names[:4])
		names_2 = list(names[4:])

		himg1 = concat_h(names_1)
		himg2 = concat_h(names_2)
		return concat_v(himg1, himg2)

	img_size = Image.open(str(path_feeder.first_fullpath)).size
	def concat_h(names: list[str], pad=h_pad)-> Image.Image:
		imim_len = len(names)
		imim = [get_numbered_img(n, number_str=n) for n in names]
		max_height = img_size[1]
		im_width = img_size[0]
		width_sum = imim_len * img_size[0]
		dst_size = (width_sum + (imim_len - 1) * pad, max_height)
		dst: Image.Image = Image.new(mode, dst_size, color=dst_bg.value)
		cur = 0
		for im in imim:
			if im:
				dst.paste(im, (cur, 0))
			cur += im_width + pad
		return dst
	def concat_v(im1: Image.Image, im2: Image.Image)-> Image.Image:
		pad = v_pad
		dst_size = (im1.width, im1.height + pad + im2.height)
		dst: Image.Image = Image.new(mode, dst_size, color=dst_bg.value)
		dst.paste(im1, (0, 0))
		dst.paste(im2, (0, im1.height + pad))
		return dst

	digit_image_feeder_S = BasicDigitImage(scale=24, line_width=6, padding=(4, 4))
	# digit_image_param_S = BASIC_DIGIT_IMAGE_PARAM_LIST[0]
	@put_number(pos=PutPos.L, digit_image_feeder=digit_image_feeder_S) #, line_width=digit_image_param_S.line_width))
	def get_numbered_img(fn: str, number_str: str)-> Image.Image | None:
		fullpath = path_feeder.dir / (fn + path_feeder.ext)
		if fullpath.exists():
			img = Image.open(fullpath)
			return img
			'''number_image, margin = get_number_image(size=number_size, nn=[int(c) for c in fn])
			offset = [s for s in (np.array(margin) + np.array(number_offset))]
			img.paste(number_image, offset)
			return img'''

	for pg, img in enumerate(get_image_blocks()):
		ct = (img.width // 2, img.height // 2) # s // 2 for s in img.size)
		ll_ww[0] = img.height // div
		ll_ww[1] = img.width // 128
		#assert img.width == IMG_SIZE[0] * 4 + H_PAD * 3
		#assert img.height == IMG_SIZE[1] * 2 + V_PAD
		drw = ImageDraw.Draw(img)
		dst = _to(pg, list(ct))
		lct = list(ct)
		dstp = lct[0] + dst[0], lct[1] + dst[1]
		month_dots(ct, drw, img.width)
		page_dots(ct, drw, pg + 1) # drw.line((*ct, *dstp), fill=(255, 255, 255), width=int(th))
		text = f"{' ' * 8}{year}-{month:02}({pg + 1}/4)"
		drw.text((*xshift(TXT_OFST, *ct), *yshift(*dstp)), text, fill=ImageFill.invert(dst_bg).value)
		# draw_digit(pg + 1, drw, offset=(10, 10), scale=30, width_ratio=8)
		#name = f"{year_month_name}-{pg + 1}.png"
		yield img #, name #.convert('L') #.save(img_dir / name, 'PNG')

from collections import namedtuple
WidthHeight = namedtuple('WidthHeight', ['width', 'height'])
def concat_8_pages(img_size: tuple[int, int], dir: Path, ext: str, names: Iterator[str], h_pad: int=0, v_pad: int=0)-> Image:
	def open_img(f: str)-> Image.Image | None:
		fullpath = dir / (f + ext)
		img =  Image.open(fullpath) if fullpath.exists() else None
		return img

	names_1 = list(names[:4])
	names_2 = list(names[4:])

	img_count = len(list(names))
	himg1 = get_concat_h(img_count, img_size, (open_img(n) for n in names_1), pad=h_pad) # (dq()))
	himg2 = get_concat_h(img_count, img_size, (open_img(n) for n in names_2), pad=h_pad) # (dq()))h4img()
	return get_concat_v(2, img_size, (himg1, himg2), pad=v_pad)

def get_img_file_names_(glob=True):
	days = calendar.monthrange(year, month)[1]
	for i in range(days):
		yield f"{'??' if glob else month}{(i + 1):02}.png"
	pad = 32 - days
	for n in range(pad):
		yield None


blank_img = Image.new('L', IMG_SIZE, (0xff,))

def open_image(dir: Path, name: str, glob=False):
	if not name:
		global file_over 
		file_over = True
		return blank_img
	if glob:
		fullpath_list = list(dir.glob(name))
		fpthslen = len(fullpath_list)
		assert fpthslen < 2
		if fpthslen == 1: #(fp:=fullpath_list[0]).exists():
			img = Image.open(fullpath_list[0]).convert('L')
			assert img.size == IMG_SIZE
			return img
		else:
			return blank_img
	else:
		assert (dir / name).exists()
		return Image.open(dir/ name)
def get_concat_h(imim_len: int, img_size: tuple[int, int], imim: Sequence[Image.Image | None], pad=0, mode='L', dst_bg=(0xff,))-> Image:
	max_height = img_size[1]
	im_width = img_size[0]
	width_sum = imim_len * img_size[0]
	dst_size = (width_sum + (imim_len - 1) * pad, max_height)
	dst: Image.Image = Image.new(mode, dst_size, dst_bg)
	cur = 0
	for im in imim:
		if im:
			dst.paste(im, (cur, 0))
		cur += im_width + pad
	return dst

def get_concat_v(imim_len: int, img_size: tuple[int, int], imim: Sequence[Image.Image | None], pad: int=0, mode='L', dst_bg=(0xff,))-> Image.Image:
	max_width = img_size[0]
	im_height = img_size[1]
	height_sum = imim_len * img_size[1]
	dst_size = (max_width, height_sum + (imim_len - 1) * pad)
	dst: Image.Image = Image.new(mode, dst_size, dst_bg)
	cur = 0
	for im in imim:
		if im:
			dst.paste(im, (0, cur))
		cur += im_height + pad
	return dst

if __name__ == '__main__':
	#save_pages()
	convert_to_pdf(layout=PdfLayout.a3lp)
	#save_arc_pages()