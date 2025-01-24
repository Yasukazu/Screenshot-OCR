from PIL import Image, ImageOps

file_over = False

def get_pages(node = '202412'):
	names = [name for name in get_img_file_names()]
	for pg in range(4):
		img = get_8(names)
		name = f"{node}-{pg + 1}.png"
		img.save(name, 'PNG')

def get_8(names):
	def dq():
		return open_img(deq(names))
	def h2img():
		return get_concat_h(dq(), dq())
	def h4img():
		return get_concat_h(h2img(), h2img())
	himg1 = h4img()
	himg2 = h4img()
	return get_concat_v(himg1, himg2)

def get_img_file_names():
	for i in range(1, 31 + 1):
		yield "12%02d.png" % i
	yield None

IMG_SIZE = (720, 1612)
def open_img(name):
	if name:
		return Image.open(name)
	global file_over 
	file_over = True
	return Image.new('RGB', IMG_SIZE)

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

def deq(lst):
	if len(lst) > 0:
		return lst.pop(0)