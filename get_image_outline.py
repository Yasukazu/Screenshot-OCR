import io
from PIL import Image, ImageFilter
from path_feeder import get_last_month_path
from pdf2image import convert_from_path

img_name = "2025-01-4p8.pdf"
img_fullpath = get_last_month_path() / img_name
imgs = convert_from_path(img_fullpath)

im = imgs[0].convert('L')

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

