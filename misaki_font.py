# Misaki font
import numpy as np
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
import sys,os
from pathlib import Path
screen_base_dir_name = os.getenv('SCREEN_BASE_DIR')
if not screen_base_dir_name:
    raise ValueError(f"{screen_base_dir_name=} is not set in env.!")
font_file_name = 'misaki_gothic.png'
font_fullpath = Path(screen_base_dir_name) / 'font' / font_file_name
if not font_fullpath.exists():
    raise ValueError(f"`{font_fullpath=}` does not exists!")
full_image = Image.open(str(font_fullpath))
ku_ten = np.array([16, 3], dtype=np.int64)
end_point = ku_ten * 8
offset = end_point - np.array([8, 8])
area = np.array([offset, end_point])
img_area = area.ravel().tolist()
font_part = full_image.crop(img_area)
font_part_4 = font_part.resize([32, 32])
font_part_4.show()