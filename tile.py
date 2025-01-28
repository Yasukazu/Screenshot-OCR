from PIL import Image
import img2pdf
import os.path, calendar

from pathlib import Path
home = Path(os.path.expanduser('~'))
year = 2024
month = 11
img_dir = home / 'Documents' / 'screen' / f'2024{month:02}'

file_over = False

node = f"{year}{month:02}"

def get_pdf():
	fullpath = img_dir / f"{node}.pdf"
	imgs = list(get_pages())
	pdf = img2pdf.convert(imgs)
	with fullpath.open('w') as wf:
		wf.write(pdf)

def get_pages():
	names = get_img_file_names() # generator
	for pg in range(4):
		img = get_8(names)
		name = f"{node}-{pg + 1}.png"
		yield img.convert('L') #.save(img_dir / name, 'PNG')

def get_8(names):
	def dq():
		return open_img(next(names))
	def h2img():
		return get_concat_h(dq(), dq())
	def h4img():
		return get_concat_h(h2img(), h2img())
	himg1 = h4img()
	himg2 = h4img()
	return get_concat_v(himg1, himg2)

def get_img_file_names():
	days = calendar.monthrange(year, month)[1]
	for i in range(days):
		yield f"{month}{(i + 1):02}.png"
	pad = 32 - days
	for n in range(pad):
		yield None

IMG_SIZE = (720, 1612)

blank_img = Image.new('L', IMG_SIZE, (0xff,))

def open_img(name):
	if not name:
		global file_over 
		file_over = True
		return blank_img
	fullpath = img_dir / name
	if fullpath.exists():
		img = Image.open(fullpath).convert('L')
		assert img.size == IMG_SIZE
		return img
	else:
		return blank_img

def get_concat_h(im1, im2, pad=20):
	dst = Image.new('RGB', (im1.width + pad + im2.width, im1.height))
	dst.paste(im1, (0, 0))
	if pad > 0:
		pad_img = Image.new('RGB', (pad, im1.height))
		dst.paste(pad_img, (im1.width, 0))
	dst.paste(im2, (im1.width + pad, 0))
	return dst

def get_concat_v(im1, im2, pad=40):
	dst = Image.new('RGB', (im1.width, im1.height + pad + im2.height))
	dst.paste(im1, (0, 0))
	if pad > 0:
		pad_img = Image.new('RGB', (im1.width, pad))
		dst.paste(pad_img, (0, im1.height))
	dst.paste(im2, (0, im1.height + pad))
	return dst

if __name__ == '__main__':
	get_pdf()