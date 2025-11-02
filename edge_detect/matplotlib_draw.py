import cv2
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from pathlib import Path

from dotenv import dotenv_values
import tomllib
from PIL import Image
cwd = Path(__file__).resolve().parent
sys.path.append(str(cwd.parent))
from set_logger import set_logger
logger = set_logger(__name__)

APP_STR = "taimee"
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
with open(filter_toml_path, 'rb') as f:
    config = tomllib.load(f)
# parent_dir = Path(__file__).resolve().parent.parent
image_path_config = config['image-path']
image_dir = Path(image_path_config['dir']).expanduser() # home dir starts with tilde(~)
if not image_dir.exists():
    raise ValueError("Error: image_dir not found: %s" % image_dir)
filename = image_path_config[APP_STR]['filename']
filename_path = Path(filename)
if '*' in filename_path.stem or '?' in filename_path.stem:
    logger.info("Trying to expand filename with wildcard: %s" % filename)
    glob_path = Path(image_dir)
    file_list = [f for f in glob_path.glob(filename) if f.is_file()]
    if len(file_list) == 0:
        raise ValueError("Error: No files found with wildcard: %s" % filename)
    logger.info("%d file found", len(file_list))
    logger.info("Files: %s", file_list)
    nth = 1
    logger.info("Choosing the %d-th file.", nth)
    filename = file_list[nth - 1].name
    logger.info("Selected file: %s", filename)
image_path = Path(image_dir) / filename
if not image_path.exists():
    raise ValueError("Error: image_path not found: %s" % image_path)
image_fullpath = image_path.resolve()
image = cv2.imread(image_fullpath) #cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
if image is None:
    raise ValueError("Error: Could not load image: %s" % image_fullpath)
from image_filter import ImageDictKey
import image_filter
app_func = getattr(image_filter, APP_STR)
if not app_func:
    raise ValueError(f"Error: Failed to load `app_func` : '{APP_STR}' in 'image_filter'")
fig, ax = plt.subplots(1, 7)
for r in range(6):
    ax[r].invert_yaxis()
    ax[r].xaxis.tick_top()
ax[0].imshow(image)
image_dict = {}
h_lines, filtered_image = app_func(image, binarize=False, cvt_color=cv2.COLOR_BGR2GRAY, image_dict=image_dict)
# mono_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
text_image = cv2.threshold(filtered_image, thresh=161, maxval=255, type=cv2.THRESH_BINARY)[1]
negative_mono_image = cv2.bitwise_not(filtered_image)
text_border_image = cv2.threshold(negative_mono_image, thresh=16, maxval=255, type=cv2.THRESH_BINARY)[1]
# negative_text_image = cv2.bitwise_not(text_image)
# negative_text_border_image = cv2.bitwise_not(text_border_image)
border_image = cv2.bitwise_and(text_border_image, text_image)

ax[1].imshow(image_dict[ImageDictKey.heading])
ax[2].imshow(image_dict[ImageDictKey.hours_from])
ax[3].imshow(image_dict[ImageDictKey.hours_to])
ax[4].imshow(image_dict[ImageDictKey.rest_hours])
ax[5].imshow(image_dict[ImageDictKey.other])
heading_image = Image.fromarray(image_dict[ImageDictKey.heading])
#from tempfile import TemporaryDirectory
from os import environ
from pytesseract import pytesseract, image_to_data, image_to_boxes, Output
from pyocr import get_available_tools, builders
ocr = get_available_tools()[1]
def ocr_lines(image: np.ndarray, from_: int = 0, to_: int | None = None, conf_min=80, tessdata_dir = "~/.local/share/tessdata/best", tesseract_cmd = '/usr/bin/tesseract'):
    pytesseract.tesseract_cmd = tesseract_cmd
    # with TemporaryDirectory() as tmpdirname: tmp_img_path = '/'.join([tmpdirname, 'test.png']) cv2.imwrite(tmp_img_path, image) text, boxes, tsv = pytesseract.run_and_get_multiple_output(tmp_img_path, extensions=['txt', 'box', 'tsv'])
    tessdata_path = Path(tessdata_dir).expanduser()
    tessdata_dir_config = r'--tessdata-dir "%s"' % str(tessdata_path)
    environ['TESSDATA_PREFIX'] = str(tessdata_path)
    data = image_to_data(image, lang="jpn", output_type=Output.DICT, config=tessdata_dir_config)
    # boxes = image_to_boxes(image, lang="jpn", output_type=Output.DICT, config=tessdata_dir_config)
    less_conf_data = [(i,data['text'][i]) for i,c in enumerate(data['conf']) if 0 < c < conf_min]
    lines = ocr.image_to_string(Image.fromarray(image), lang="jpn", builder=builders.LineBoxBuilder())
    return [t.content.replace(' ','') for t in (lines[from_:to_] if to_ is not None else lines[from_:])], less_conf_data if len(less_conf_data) > 0 else None, data if len(less_conf_data) > 0 else None
lines_to_dict = {
    ImageDictKey.heading: -1,
    ImageDictKey.hours_from: None,
    ImageDictKey.hours_to: None,
    ImageDictKey.rest_hours: None,
    ImageDictKey.other: None
}
for key, to_ in lines_to_dict.items():
    lines, confs, data= ocr_lines((image_dict[key]), to_=to_) # Image.fromarray
    text = '\t'.join(lines)
    print(f"{key.name}: {text}")
    print(confs) if confs else None
    print(data) if data else None
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
