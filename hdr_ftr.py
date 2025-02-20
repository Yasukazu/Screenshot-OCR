import sys
from PIL import Image, ImageFilter, ImageSequence
from path_feeder import get_last_month_path

img_name = "2025-01-img32.tif" # 2940x3264
img_fullpath = get_last_month_path() / img_name
imgs = Image.open(img_fullpath)
for i, page in enumerate(ImageSequence.Iterator(imgs)):
    im = page.convert('L')
    break
hdr_ftr_w = im.size[0]
hdr_ftr_h = im.size[1] // 32
H_F_BG = (0xff,)
hdr = Image.new('L', (hdr_ftr_w, hdr_ftr_h), color=H_F_BG)
ftr = Image.new('L', (hdr_ftr_w, hdr_ftr_h), color=H_F_BG)
from tile import get_concat_v
h_im_f = get_concat_v(hdr, im, ftr, pad=0)
h_im_f.show()
sys.exit(0)
# Detect edges and save
edges = im.filter(ImageFilter.FIND_EDGES)
h_size = edges.size[1]
corner_img = edges.crop((0, 0, 2, h_size / 4))
import numpy as np
corner_array = np.array(corner_img)
top_edge = 0
for i, line in enumerate(corner_array[1:]):
    if line[1] > 0:
        top_edge = i
        break
print(top_edge) # 249

