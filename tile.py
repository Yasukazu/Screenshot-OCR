from PIL import Image, ImageDraw
import img2pdf
import os.path, calendar
import ttl
from pathlib import Path
home = Path(os.path.expanduser('~'))
year = 2025
month = 1
img_dir = home / 'Documents' / 'screen' / f'{year}{month:02}'

IMG_SIZE = (720, 1612)
H_PAD = 20
V_PAD = 40

file_over = False

node = f"{year}-{month:02}"

a4inpt = (img2pdf.mm_to_pt(210),img2pdf.mm_to_pt(297))
a3inpt = (img2pdf.mm_to_pt(297),img2pdf.mm_to_pt(420))


def paged_png_feeder():
	for p in range(4):
		yield img_dir / f"{year}-{month:02}-{p + 1}.png"

def convert_to_pdf(names=paged_png_feeder(),
	fullpath = img_dir / f"{node}.pdf", layout=a4inpt):
	layout_fun = img2pdf.get_layout_fun(layout)
	with open(fullpath,"wb") as f:
		#(img, name) in get_pages()]
		for name in names:
			assert Path(name).exists()
		f.write(img2pdf.convert(layout_fun=layout_fun, *names, rotation=img2pdf.Rotation.ifvalid))

def save_pages_as_pdf():
	fullpath = img_dir / f"{node}.pdf"
	imges = list(get_quad_pages())
	imges[0].save(fullpath, "PDF" ,resolution=100.0, save_all=True, append_imges=imges[1:])

def save_pages():
	for page, name in get_quad_pages():
		fullpath = img_dir / name
		page.save(fullpath, 'PNG')

TXT_OFST = 80
from load_csv_7 import draw_num
def get_quad_pages(div=64, th=H_PAD // 2):
	drc_tbl = [(0, -1), (1, 0), (0, 1), (-1, 0)]
	ll_ww = [0, 0]
	def _to(n, xy):
		ll, ww = ll_ww[0], ll_ww[1]
		x = xy[0]
		y = xy[1]
		drc = drc_tbl[n]
		return x + ll * drc[0], y + ll * drc[1]
	names = list(get_quad_png_file_names()) # generator
	def xshift(offset, *xy):
		return xy[0] + offset, xy[1]
	def yshift(*xy):
		x, y = xy[0], xy[1]
		return x, y - (V_PAD - 8)
	fill_white = (255,255,255)
	def page_dots(ct, drw, n):
		ll, ww = ll_ww[0], ll_ww[1]
		op = list(ct)
		for i in range(n):
			dp = op[0] - ww, op[1]
			drw.line((*op, *dp), fill_white, width=int(th))
			op[0] -= 2 * ww
	def month_dots(ct, drw, img_w):
		ttl.set_pit_len(img_w // 32)
		ttl.set_org(*(ct[0] + (img_w // 32), ct[1]))
		for frm, to in ttl.plot(month, ttl.Direc.RT):
			drw.line((frm[0], frm[1], to[0], to[1]), fill_white, width=int(th))

	for pg, name in enumerate(names): #range(4):
		img = Image.open(name) # get_quad_page(names)
		ct = (img.width // 2, img.height // 2) # s // 2 for s in img.size)
		ll_ww[0] = img.height // div
		ll_ww[1] = img.width // 128
		assert img.width == IMG_SIZE[0] * 4 + H_PAD * 3
		assert img.height == IMG_SIZE[1] * 2 + V_PAD
		drw = ImageDraw.Draw(img)
		dst = _to(pg, list(ct))
		lct = list(ct)
		dstp = lct[0] + dst[0], lct[1] + dst[1]
		month_dots(ct, drw, img.width)
		page_dots(ct, drw, pg + 1) # drw.line((*ct, *dstp), fill=(255, 255, 255), width=int(th))
		text = f"{' ' * 8}{year}-{month:02}({pg + 1}/4)"
		drw.text((*xshift(TXT_OFST, *ct), *yshift(*dstp)), text, 'white')
		draw_num(pg + 1, drw, offset=(10, 10), scale=30, width=8)
		name = f"{node}-{pg + 1}.png"
		yield img, name #.convert('L') #.save(img_dir / name, 'PNG')

def get_quad_page(names):
	def dq():
		return Image.open(next(names))
	def h2img():
		return get_concat_h(dq(), dq())
	def h4img():
		return get_concat_h(h2img(), h2img())
	himg1 = h4img()
	himg2 = h4img()
	return get_concat_v(himg1, himg2)

def get_img_file_names_(glob=True):
	days = calendar.monthrange(year, month)[1]
	for i in range(days):
		yield f"{'??' if glob else month}{(i + 1):02}.png"
	pad = 32 - days
	for n in range(pad):
		yield None

from path_feeder import path_feeder, FileExt
def get_quad_png_file_names():
	for path in path_feeder(input_type=FileExt.QPNG):
		yield path

blank_img = Image.new('L', IMG_SIZE, (0xff,))

def open_img(name,glob=True):
	if not name:
		global file_over 
		file_over = True
		return blank_img
	if glob:
		fullpath_list = list(img_dir.glob(name))
		fpthslen = len(fullpath_list)
		assert fpthslen < 2
		if fpthslen == 1: #(fp:=fullpath_list[0]).exists():
			img = Image.open(fullpath_list[0]).convert('L')
			assert img.size == IMG_SIZE
			return img
		else:
			return blank_img
	else:
		return Image.open(name)

def get_concat_h(im1, im2, pad=H_PAD):
	dst = Image.new('RGB', (im1.width + pad + im2.width, im1.height))
	dst.paste(im1, (0, 0))
	if pad > 0:
		pad_img = Image.new('RGB', (pad, im1.height))
		dst.paste(pad_img, (im1.width, 0))
	dst.paste(im2, (im1.width + pad, 0))
	return dst

def get_concat_v(im1, im2, pad=V_PAD):
	dst = Image.new('RGB', (im1.width, im1.height + pad + im2.height))
	dst.paste(im1, (0, 0))
	if pad > 0:
		pad_img = Image.new('RGB', (im1.width, pad))
		dst.paste(pad_img, (0, im1.height))
	dst.paste(im2, (0, im1.height + pad))
	return dst

if __name__ == '__main__':
	save_pages()
	#convert_to_pdf()