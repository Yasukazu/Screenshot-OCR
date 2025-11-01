import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from dotenv import dotenv_values
import tomllib
from PIL import Image

FILTER_TOML_PATH_STR = "FILTER_TOML_PATH"
filter_toml_path = None 
if not filter_toml_path:
    env_file = Path('.env')
    if env_file.exists():
        with env_file.open('r') as f:
            filter_toml_path = dotenv_values(stream=f)[FILTER_TOML_PATH_STR]
    else:
        filter_toml_path = os.environ[FILTER_TOML_PATH_STR]
if not filter_toml_path:
    raise ValueError("Error: `filter_toml_path` is empty!")
config_file = Path(filter_toml_path)
with config_file.open('rb') as f:
    config = tomllib.load(f)
# parent_dir = Path(__file__).resolve().parent.parent
image_path_config = config['image-path']
image_dir = Path(image_path_config['dir'])
if not image_dir.exists():
    raise ValueError("Error: image_dir not found: %s" % image_dir)
filename = image_path_config['filename']
image_path = Path(image_dir) / filename
if not image_path.exists():
    raise ValueError("Error: image_path not found: %s" % image_path)
image_fullpath = image_path.resolve()
image = cv2.imread(image_fullpath) #cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
if image is None:
    raise ValueError("Error: Could not load image: %s" % image_fullpath)
from image_filter import taimee, ImageDictKey
image_dict = {}
h_lines, filtered_image = taimee(image, binarize=False, cvt_color=cv2.COLOR_BGR2GRAY, image_dict=image_dict)
# mono_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
text_image = cv2.threshold(filtered_image, thresh=161, maxval=255, type=cv2.THRESH_BINARY)[1]
negative_mono_image = cv2.bitwise_not(filtered_image)
text_border_image = cv2.threshold(negative_mono_image, thresh=16, maxval=255, type=cv2.THRESH_BINARY)[1]
# negative_text_image = cv2.bitwise_not(text_image)
# negative_text_border_image = cv2.bitwise_not(text_border_image)
border_image = cv2.bitwise_and(text_border_image, text_image)
fig, ax = plt.subplots(1, 7)
for r in range(6):
    ax[r].invert_yaxis()
    ax[r].xaxis.tick_top()
ax[0].imshow(filtered_image)
ax[1].imshow(image_dict[ImageDictKey.heading])
ax[2].imshow(image_dict[ImageDictKey.hours_from])
ax[3].imshow(image_dict[ImageDictKey.hours_to])
ax[4].imshow(image_dict[ImageDictKey.rest_hours])
ax[5].imshow(image_dict[ImageDictKey.other])
heading_image = Image.fromarray(image_dict[ImageDictKey.heading])
from pyocr import get_available_tools, builders
ocr = get_available_tools()[1]
def get_lines(heading_image: Image, from_: int = 0, to_: int | None = None):
    heading_lines = ocr.image_to_string(heading_image, lang="jpn", builder=builders.LineBoxBuilder())
    return [t.content.replace(' ','') for t in (heading_lines[from_:to_] if to_ is not None else heading_lines[from_:])]
lines_to_dict = {
    ImageDictKey.heading: -1,
    ImageDictKey.hours_from: None,
    ImageDictKey.hours_to: None,
    ImageDictKey.rest_hours: None,
    ImageDictKey.other: None
}
for key, to_ in lines_to_dict.items():
    lines = get_lines(Image.fromarray(image_dict[key]), to_=to_)
    text = '\t'.join(lines)
    print(f"{key.name}: {text}")

r = np.array(negative_mono_image)[:, :].flatten()
bins_range = range(0, 257, 8)
xtics_range = range(0, 257, 32)
# fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)

ax[6].hist(r, bins=bins_range, color='r')
plt.setp((ax[6],), xticks=xtics_range, xlim=(0, 256))
ax[6].grid(True)

def press(event):
    print(event.key)
    if event.key == 'q':
        plt.close()
fig.canvas.mpl_connect('key_press_event', press)
plt.show()
print(h_lines)
# print(cut_x, cut_height)
# print(y_coords[last + 1:])
