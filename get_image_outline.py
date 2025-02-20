from PIL import Image, ImageFilter
from path_feeder import get_last_month_path
from pdf2image import convert_from_path

img_name = "2025-01-4p8.pdf"
img_fullpath = get_last_month_path() / img_name
imgs = convert_from_path(img_fullpath)

im = imgs[0] # Image.open(img_fullpath).convert('L')

# Detect edges and save
edges = im.filter(ImageFilter.FIND_EDGES)
out_name = img_name + '-1-edges.png'
edges.save(out_name)
