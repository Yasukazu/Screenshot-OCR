import cv2
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from pathlib import Path
from io import StringIO

from dotenv import dotenv_values
import tomllib
from PIL import Image
cwd = Path(__file__).resolve().parent
sys.path.append(str(cwd.parent))
from set_logger import set_logger
logger = set_logger(__name__)

APP_STR = "taimee"
FILTER_TOML_PATH_STR = "FILTER_TOML_PATH"
filter_toml_path = getattr(os.environ, FILTER_TOML_PATH_STR, None) 
ENV_FILE_NAME = ".env"
env_file = Path(ENV_FILE_NAME)
try: # if env_file.exists():
	with env_file.open('r') as f:
		env_filter_toml_path = dotenv_values(stream=f)[FILTER_TOML_PATH_STR]
		if env_filter_toml_path:
			filter_toml_path = env_filter_toml_path
except KeyError:
	logger.warning("KeyError: '%s' not found in %s", FILTER_TOML_PATH_STR, ENV_FILE_NAME)
except FileNotFoundError:
	logger.warning("FileNotFoundError: '%s' not found", ENV_FILE_NAME)
if not filter_toml_path:
	raise ValueError("Error: failed to load 'filter_toml_path'!")
try:
	with open(filter_toml_path, 'rb') as f:
		filter_config = tomllib.load(f)

	image_path_config = filter_config['image-path']
	image_dir = Path(image_path_config['dir']).expanduser() # home dir starts with tilde(~)
	if not image_dir.exists():
		raise ValueError("Error: image_dir not found: %s" % image_dir)
	image_config_filename = image_path_config[APP_STR]['filename']
	filename_path = Path(image_config_filename)
	if '*' in filename_path.stem or '?' in filename_path.stem:
		logger.info("Trying to expand filename with wildcard: %s" % image_config_filename)
		glob_path = Path(image_dir)
		file_list = [f for f in glob_path.glob(image_config_filename) if f.is_file()]
		if len(file_list) == 0:
			raise ValueError("Error: No files found with wildcard: %s" % image_config_filename)
		logger.info("%d file found", len(file_list))
		logger.info("Files: %s", file_list)
		nth = 1
		logger.info("Choosing the %d-th file.", nth)
		image_config_filename = file_list[nth - 1].name
		logger.info("Selected file: %s", image_config_filename)
	image_path = Path(image_dir) / image_config_filename
	if not image_path.exists():
		raise ValueError("Error: image_path not found: %s" % image_path)
	image_fullpath = image_path.resolve()
except KeyError as err:
	raise ValueError("Error: key not found!\n%s" % err)
except FileNotFoundError as err:
	raise ValueError("Error: file not found!\n%s" % err)
except tomllib.TOMLDecodeError as err:
	raise ValueError("Error: failed to TOML decode!\n%s" % err)
image = cv2.imread(str(image_fullpath)) #cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
if image is None:
	raise ValueError("Error: Could not load image: %s" % image_fullpath)
from image_filter import ImageDictKey, taimee, BinaryImage, TaimeeFilter, ImageFilterParam, ImageAreaName
from typing import Any
# import image_filter
'''title_window = 'Binary Iimage'
cv2.namedWindow(title_window)
binary_image = BinaryImage(image, binarize=True)
g_threshold = 237
def on_trackbar(val):
	global g_threshold
	g_threshold = val
	b_image = binary_image.bin_image(thresh_val=g_threshold)
	cv2.imshow('Binary Image', b_image)
cv2.createTrackbar('Threshold', title_window, g_threshold, 255, on_trackbar)
on_trackbar(g_threshold)
cv2.waitKey(0)
b_image = binary_image.bin_image(thresh_val=g_threshold)
SUBPLOT_SIZE = 2
fig, ax = plt.subplots(SUBPLOT_SIZE, 1)#, figsize=(10, 4*SUBPLOT_SIZE))
for r in range(SUBPLOT_SIZE):
	ax[r].invert_yaxis()
	ax[r].xaxis.tick_top()
	ax[r].set_title(f"Row {r+1}")
ax[0].imshow(b_image, cmap='gray')'''
filter_param_dict: dict[ImageFilterParam, Any] = {}
taimee_filter = TaimeeFilter(given_image=image, params=filter_param_dict)
heading_param_dict = {}
#heading_ypos, heading_height, heading_xpos, heading_width 
t_leading_y = taimee_filter.leading_y
heading_param = taimee_filter.extract_heading(heading_param_dict)
filter_param_dict |= heading_param_dict

	# wf.write(','.join([f"'{k.name}':{list(v)}" for k, v in filter_param_dict.items()]) + '}\n')
fig, ax = plt.subplots(1, 2)
ax[1].imshow(taimee_filter.bin_image[t_leading_y:t_leading_y+heading_param.height, heading_param.xpos:], cmap='gray')
plt.show()
from io import StringIO
wf = StringIO()
# wf.write("[taimee]\n")
wf.write("[heading-param.taimee]\n")
heading_param.to_toml(wf)
toml_str = wf.getvalue()
wf.close()
toml_filename = 'taimee-filter.toml'
with open(toml_filename, 'w') as wf:
	wf.write(toml_str)
from tomllib import load
with open(toml_filename, 'rb') as rf:
	toml_dict = load(rf)
heading_image_area = ImageAreaName(**toml_dict['heading-param']['taimee'])
print(toml_dict)
image_filter_apps = {'taimee': taimee}
app_func = image_filter_apps[APP_STR]
if not app_func:
	raise ValueError(f"Error: Failed to load `app_func` : '{APP_STR}' in 'image_filter'")
SUBPLOT_SIZE = 6
fig, ax = plt.subplots(SUBPLOT_SIZE, 1)
for r in range(SUBPLOT_SIZE):
	ax[r].invert_yaxis()
	ax[r].xaxis.tick_top()
	ax[r].set_title(f"Row {r}")
image_dict = {}
h_lines, filtered_image = app_func(image, single=False, cvt_color=cv2.COLOR_BGR2GRAY, image_dict=image_dict, binarize=True)
# mono_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
binary_image = cv2.threshold(filtered_image, thresh=150, maxval=255, type=cv2.THRESH_BINARY)[1]
# ax[0].imshow(image)
ax[0].imshow(binary_image)
# negative_mono_image = cv2.bitwise_not(filtered_image)
# text_border_image = cv2.threshold(negative_mono_image, thresh=16, maxval=255, type=cv2.THRESH_BINARY)[1]
# negative_text_image = cv2.bitwise_not(text_image)
# negative_text_border_image = cv2.bitwise_not(text_border_image)
# border_image = cv2.bitwise_and(text_border_image, text_image)

ax[1].imshow(image_dict[ImageDictKey.heading])
ax[2].imshow(image_dict[ImageDictKey.shift_start])
ax[3].imshow(image_dict[ImageDictKey.shift_end])
ax[4].imshow(image_dict[ImageDictKey.break_time])
ax[5].imshow(image_dict[ImageDictKey.payslip])
heading_image = Image.fromarray(image_dict[ImageAreaKey.heading])
#from tempfile import TemporaryDirectory
from os import environ
from pytesseract import pytesseract, image_to_data, image_to_boxes, Output

class TesseractOCR:
	def __init__(self, tessdata_dir = "~/.local/share/tessdata/fast", tesseract_cmd = '/usr/bin/tesseract', psm_value = 6):
		self.tessdata_dir = tessdata_dir
		self.tesseract_cmd = tesseract_cmd
		self.conf_min=80
		pytesseract.tesseract_cmd = self.tesseract_cmd
	# with TemporaryDirectory() as tmpdirname: tmp_img_path = '/'.join([tmpdirname, 'test.png']) cv2.imwrite(tmp_img_path, image) text, boxes, tsv = pytesseract.run_and_get_multiple_output(tmp_img_path, extensions=['txt', 'box', 'tsv'])
		self.psm_value = psm_value

	@property
	def psm_config(self):
		return "--psm %d" % self.psm_value

	@property
	def tessdata_dir_config(self):
		tessdata_path = Path(self.tessdata_dir).expanduser()
		return f'--tessdata-dir {tessdata_path}'

	def exec(self, image: np.ndarray, lang="jpn",output_type=Output.DICT):
		return image_to_data(image, lang=lang, output_type=output_type, config=' '.join([self.tessdata_dir_config, self.psm_config]))

from pyocr import get_available_tools, builders
ocr = get_available_tools()[1]
def ocr_lines(image: np.ndarray, from_: int = 0, to_: int | None = None, conf_min=80, tessdata_dir = "~/.local/share/tessdata/fast", tesseract_cmd = '/usr/bin/tesseract'):
	pytesseract.tesseract_cmd = tesseract_cmd
	# with TemporaryDirectory() as tmpdirname: tmp_img_path = '/'.join([tmpdirname, 'test.png']) cv2.imwrite(tmp_img_path, image) text, boxes, tsv = pytesseract.run_and_get_multiple_output(tmp_img_path, extensions=['txt', 'box', 'tsv'])
	tessdata_path = Path(tessdata_dir).expanduser()
	tessdata_dir_config = r'--tessdata-dir "%s"' % str(tessdata_path)
	psm_value = 6
	psm_config = "--psm %d" % psm_value
	environ['TESSDATA_PREFIX'] = str(tessdata_path)
	data = image_to_data(image, lang="jpn", output_type=Output.DICT, config=tessdata_dir_config + ' ' + psm_config)
	# boxes = image_to_boxes(image, lang="jpn", output_type=Output.DICT, config=tessdata_dir_config)
	less_conf_data = [(i,data['text'][i]) for i,c in enumerate(data['conf']) if 0 < c < conf_min]
	lines = ocr.image_to_string(Image.fromarray(image), lang="jpn", builder=builders.LineBoxBuilder(tesseract_layout=psm_value))
	return [t.content.replace(' ','') for t in (lines[from_:to_] if to_ is not None else lines[from_:])], less_conf_data if len(less_conf_data) > 0 else None, data if len(less_conf_data) > 0 else None
lines_to_dict = {
	ImageAreaKey.heading: -1,
	ImageAreaKey.shift_start: None,
	ImageAreaKey.shift_end: None,
	ImageAreaKey.break_time: None,
	ImageAreaKey.payslip: None
}
with StringIO() as f:
	for key, to_ in lines_to_dict.items():
		lines, confs, data= ocr_lines((image_dict[key]), to_=to_) # Image.fromarray
		text = '\t'.join(lines)
		print(f"{key.name}: {text}")
		print(f"{key.name}: {text}\n", file=f)
		print(confs, file=f) if confs else None
		print(data, file=f) if data else None
		f.write('\n')
	f.seek(0)
	conf_data = f.read()
conf_fullpath = image_fullpath.parent / (image_fullpath.stem + '.conf')
with conf_fullpath.open('w') as wf:
	wf.write(conf_data)
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
