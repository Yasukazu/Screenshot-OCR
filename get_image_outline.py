from PIL import Image, ImageFilter
from path_feeder import get_last_month_path
from pdf2image import convert_from_path

img_name = "2025-01-4p8.pdf"
imag_fullmath = get_last_month_path / img_name
im = Image.open('star.png').convert('L')

# Detect edges and save
edges = im.filter(ImageFilter.FIND_EDGES)
edges.save('DEBUG-edges.png')

from path_feeder import get_last_month_path