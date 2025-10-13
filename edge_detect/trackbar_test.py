from typing import NamedTuple, Optional
from tomllib import load as load_toml_file
from math import sqrt, floor, dist
from typing import Sequence
from sys import argv, exit, stdout

import cv2 
import cv2 as cv
import numpy as np 
from numpy import ndarray
import matplotlib.pyplot as plt
import tomlkit
from tomlkit.toml_file import TOMLFile
from tomlkit import TOMLDocument	

from logging import getLogger, INFO, StreamHandler, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)
stdout_handler = StreamHandler(stdout)
stdout_handler.setLevel(DEBUG)
logger.addHandler(stdout_handler)

from pathlib import Path
from typing import NamedTuple

class Rect(NamedTuple):
	x: int
	y: int
	w: int
	h: int

from dataclasses import dataclass
@dataclass
class CropRatio:
	h: float
	w: float
	def as_size(self, image_h, image_w):
		return int(image_h * self.h), int(image_w * self.w)
	def as_shape(self, image_h, image_w):
		return tuple(reversed(self.as_size(image_h, image_w)))

# def get_aspect_ratio(cont: ndarray) -> float: rect = cv2.minAreaRect(cont)

def main(filepath: Path, threshold_ratio=0.5, BGR='B', max_rect_aspect=10.0,
	vertical_crop_ratio=1.0, manual_mask=False): # title_window = 'Tracbar test', config_dir='Data'):
	logger.debug("%f threshold_ratio:level %f", threshold_ratio, threshold_ratio * 255)
	logger.debug("%f max_rect_aspect", max_rect_aspect)
	logger.info("%f vartical_crop_ratio", vertical_crop_ratio)
	image = cv2.imread(str(filepath))
	if image is None:
		logger.error("Failed to load image: %s", filepath)
		raise ValueError("Failed to load image: %s", filepath)
	image_h, image_w, _ = image.shape 
	logger.debug("image_h: %d, image_w: %d", image_h, image_w)
	mask_ratio = get_image_mask(image, filepath, manual=manual_mask)
	logger.debug("mask: %s", mask_ratio)
	# mask_image = np.full((*mask_ratio.as_size(image_h, image_w), 3), (255, 255, 255), dtype=np.uint8)
	masked_image = image.copy() # np.bitwise_or(image, mask_image)
	cv2.rectangle(masked_image, (0, 0), mask_ratio.as_shape(image_h, image_w), (255, 255, 255), cv2.FILLED)
	cv2.imshow('Masked', masked_image)
	cv2.waitKey(0)
	# custom_config = '--psm 6'#--oem 3 
	import pytesseract
	data = pytesseract.image_to_data(masked_image, lang='jpn') #, config='--psm 4')
	import csv
	import pandas
	import io
	df = pandas.read_csv(io.StringIO(data), sep='\t')
	tsv = csv.DictReader(data.splitlines(), delimiter='\t')
	tsv_key_names = '''level: レイアウト解析結果のレベル（ページ、ブロック、段落など）
	page_num: ページ番号
	block_num: ブロック番号
	par_num: 段落番号
	line_num: 行番号
	word_num: 単語番号
	left, top, width, height: 文字を囲むボックスの左上角の座標と幅、高さ
	conf: 確からしさ
	text: 認識結果の文字列'''
	tsv_key_list = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text']
	for row in tsv:
		for key in tsv_key_list:
			print(f"{key}: {row[key]}")
	exit(0)
"""
		if config_filename.exists():
			logger.info("Start to load config from a file: %s", config_filename)
			with open(config_filename, 'r') as f:
				config_doc = tomlkit.load(f)
			logger.info("Loaded %d config items.", len(config_doc))
			h_crop_key = 'h_crop_r'
			h_crop_to_crop_ratio = {h_crop_key: crop_ratio}
			config_doc |= h_crop_to_crop_ratio
			breakpoint()
			try: # config_doc.update(image_area_key, {h_crop_key: crop_ratio})
				image_area_tbl = config_doc[image_area_key]
				logger.debug('config_doc[%s] exists as %s', image_area_key, image_area_tbl)
				image_area_tbl[h_crop_key] = crop_ratio
			except KeyError as k:
				logger.debug("config_doc key error[image-area]: %s", k.args[0])
				config_doc.add(image_area_key, {h_crop_key: crop_ratio})
				# image_area_tbl = tomlkit.table()
				# image_area_tbl.add(image_area_key, {h_crop_key: crop_ratio})
			# image_area_tbl.update()
			try:
				# config_doc[f'image-area.{filename.stem}']['h_crop_r'] = crop_ratio
				# logger.debug("config_doc: %s", config_doc[f'image-area.{filename.stem}'])
				with open(config_filename, 'w') as f:
					tomlkit.dump(config_doc, f)
					logger.info("config is saved for crop_ratio in [%s] as file: '%s'", image_area_key, config_filename)
			except KeyError as err:
				logger.error("config_doc key error[image-area]: %s", err)
				raise
		else:
			config_doc = tomlkit.table()
			crop_tbl = tomlkit.table()
			crop_tbl.add('h_crop_r', crop_ratio)
			cfg_tbl = tomlkit.table()
			cfg_tbl.add(f'image-area.{filepath.stem}', crop_tbl)
			with open(config_filename, 'w') as f:
				tomlkit.dump(cfg_tbl, f)
			logger.info("Config file is newly created as: %s", config_filename)
			# logger.info("Saved config as: %s", config_filename)

		exit(0)
	if vertical_crop_ratio < 1.0:
		image_h = floor(image_h * vertical_crop_ratio)
		image = image[:image_h]
		logger.info("image_h is cropped by: %f", vertical_crop_ratio)
	# image_min = min([image_h, image_w])

	src_f = image.astype(np.float64)
	# Convert to grayscale 
	luminosity_result = np.zeros((image_h, image_w), np.uint8)
	# ルミナンス法（Luminosity Method）
	bgr_pos = 'BGR'.index(BGR)
	for i in range(image_h):
		for j in range(image_w):
			#blue = src_f[i, j, 0]
			#green = src_f[i, j, 1]
			#red = src_f[i, j, 2]
			luminosity_result[i, j] = src_f[i, j, bgr_pos] # 0.02126*red + 0.5152*green + 0.722*blue
	# cv2.imshow('Luminosity result', luminosity_result)
	# cv2.waitKey(0) 
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
	
	# Apply Gaussian blur to reduce noise and improve edge detection 
	blurred = cv2.GaussianBlur(luminosity_result, (17, 17), 3) # medianBlur
	# Show histogram
	histSize = 256
	histRange = [0, 256]
	histogram = cv2.calcHist([blurred], [0], None, [histSize], histRange, accumulate=False)
	# ヒストグラムの可視化
	hist_w = 512
	hist_h = 400
	bin_w = int(round( hist_w/histSize ))
	histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
	cv2.normalize(histogram, histogram, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

	for i in range(1, histSize):
		cv2.line(histImage, ( bin_w*(i-1), hist_h - int(histogram[i-1]) ),
				( bin_w*(i), hist_h - int(histogram[i]) ),
				( 255, 0, 0), thickness=2)

	cv2.imshow('Source image', image)
	cv2.imshow('Histogram', histImage)
	# cv2.waitKey()

	plt.rcParams["figure.figsize"] = [12,3.8]                        # 表示領域のアスペクト比を設定
	plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.9) # 余白を設定
	plt.subplot(121)                                                 # 1行2列の1番目の領域にプロットを設定
	plt.imshow(image, cmap='gray')                                   # 画像をグレースケールで表示
	plt.axis("off")                                                  # 軸目盛、軸ラベルを消す
	plt.subplot(122)                                                 # 1行2列の2番目の領域にプロットを設定
	plt.plot(histogram)                                              # ヒストグラムのグラフを表示
	plt.xlabel('Brightness')                                         # x軸ラベル(明度)
	plt.ylabel('Count')                                              # y軸ラベル(画素の数)
	plt.show()
	# cv2.imshow('Blur', blurred)

	# Sobel filter
	sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0) # X
	sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1) # Y

	# 立下りエッジ（白から黒へ変化する部分）がマイナスになるため絶対値を取る
	# alphaの値は画像表示に合わせて倍率調整
	sobel_x = cv2.convertScaleAbs(sobel_x, alpha = 0.5)
	sobel_y = cv2.convertScaleAbs(sobel_y, alpha = 0.5)
	# X方向とY方向を足し合わせる
	sobel_xy = cv2.add(sobel_x, sobel_y)
	cv2.imshow('Sobel filtered', sobel_xy)
	cv2.waitKey(0) 
	# Binarize
	ret, binarized = cv2.threshold(blurred, int(threshold_ratio * 255), 255, cv2.THRESH_BINARY)
	# binarized = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	cv2.imshow('Binarized', binarized)
	# cv2.waitKey(0)
	# exit(0)
	
	# Perform edge detection 
	canny_edges = cv2.Canny(binarized, 50, 150) 
	cv2.imshow('Cunny edged', canny_edges)
	cv2.waitKey(0) 
	# Find contours in the edged image 
	contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, #TREE, 
	cv2.CHAIN_APPROX_SIMPLE) 
	if not len(contours):
		raise ValueError("No contours detected!")
	logger.info("%s contours detected.", len(contours))
	#limit = 4 # len(contours) // 32
	#limited_cont = []
	# aspect_limited_contours = [ct for ct in contours if cv2.con]
	sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True) #[:limit] # (limit // 2):limit] 
	# if len(ct) == 4: limited_cont.append(ct) if len(limited_cont) >= limit: break
	# logger.info("%s contours limited.", len(limited_cont))
	# image_cutoff = floor(image_min * cutoff) # / 16) or 4
	# Loop over the contours 
	# contour_limit_rate = 0.9
	contour_limit = 80 # int(len(sorted_contours) * contour_limit_rate)
	logger.info("contour_limit: %d", contour_limit)
	added = 0
	# bounding_rect_contours = [cv2.boundingRect(contour) for contour in sorted_contours[:limit]] 
	bounding_rect_contours = []
	for contour in sorted_contours[:contour_limit]: 
		rect = Rect(*cv2.boundingRect(contour))
		aspect_ratio = max(rect.w / rect.h, rect.h / rect.w)
		if aspect_ratio < max_rect_aspect:
			logger.debug("rect aspect_ratio: %f", aspect_ratio)
			bounding_rect_contours.append(contour)
		# Approximate the contour to a polygon 
		# epsilon = 0.02 * cv2.arcLength(contour, True) 
		# approx_cont = cv2.approxPolyDP(contour, epsilon, True) 
		# Check if the approximated contour has 4 points (rectangle) 
		# if len(approx_cont) == 4:
			# min_dist = get_min_distance(approx_cont)
			# if min_dist > image_cutoff: 
				# Draw the rectangle on the original image 
			# cv2.drawContours(image, contours, -1, (0xff, 0, 0), 2) 
			cv2.rectangle(image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0xef, 0xef, 0xef), 4)
			cv2.drawContours(image, [contour], -1, (0xff, 0, 0), 2) 
			cv2.imshow('Detected bounding rectangles', image) 
			cv2.waitKey(20) 
			added += 1
				# if added >= limit: break
	
	# Display the result 
	cv2.waitKey(0) 
"""

def get_image_mask(image: ndarray,
	trackbar_slider_max = 100,
	title_window = 'Image mask', 
	crop_ratio: Optional[CropRatio] = None,
	config_doc: Optional[TOMLDocument] = None,
	app_name: str, # application name of image (stem.rsplit('_', 1)[0] of "img1.screenshot.png"): stem_app.ext i.e. postfix of stem separated by '_'
	manual = False) -> CropRatio: 
	image_h, image_w = image.shape[:2]
	image_area_key = f'image-area.{app_name}'
	if not crop_ratio:
		if config_doc is not None:
			image_area_config = config_doc.get(image_area_key)
			if image_area_config is not None:
				crop_ratio = CropRatio(image_area_config.get('h_crop_r') or 0.0, image_area_config.get('w_crop_r') or 0.0)
			else:
				crop_ratio = CropRatio(0.0, 0.0)
		else:
			crop_ratio = CropRatio(0.0, 0.0)
	if not manual:
		return crop_ratio
	def show_rect_image():
		image2 = image.copy()
		cv.rectangle(image2, (0, 0), (int(crop_ratio.w * image_w), int(crop_ratio.h * image_h)), (255, 0, 0), 4)
		cv.imshow(title_window, image2)
	def on_trackbar(slider_pos: int, h_w: str):
		assert 0 <= slider_pos <= trackbar_slider_max
		match h_w:
			case 'h':
				crop_ratio.h = slider_pos / trackbar_slider_max
			case 'w':
				crop_ratio.w = slider_pos / trackbar_slider_max
			case _:
				raise ValueError('Undefined h_w: %s', h_w)
		show_rect_image()
	def on_trackbar_h(slider_pos: int):
		on_trackbar(slider_pos, h_w='h')
	def on_trackbar_w(slider_pos: int):
		on_trackbar(slider_pos, h_w='w')

	cv.namedWindow(title_window)
	trackbar_name = "Crop {HW} ratio percent [max: {max}] | Hit: 'q' to exit; 's' to save config" # % trackbar_slider_max
	cv.createTrackbar(trackbar_name.format(HW='H', max=trackbar_slider_max), title_window , int(crop_ratio.h * trackbar_slider_max), trackbar_slider_max, on_trackbar_h)
	cv.createTrackbar(trackbar_name.format(HW='W', max=trackbar_slider_max), title_window , int(crop_ratio.w * trackbar_slider_max), trackbar_slider_max, on_trackbar_w)
	# on_trackbar(trackbar_slider_max)
	image2 = image.copy()
	cv.rectangle(image2, (0, 0), (int(crop_ratio.w * image_w), int(crop_ratio.h * image_h)), (0, 0, 255), 4)
	cv.imshow(title_window, image2)
	key = cv2.waitKey(0)
	return crop_ratio
	"""if key in (ord('q'), ord('Q')):
		logger.debug("Terminated by user. crop_ratios[h/w]: %s", crop_ratio)
		return crop_ratio
	if key in (ord('s'), ord('S')):
		if (cfg_crop_ratio_h is not None and cfg_crop_ratio_h == crop_ratio.h) and (cfg_crop_ratio_w is not None and cfg_crop_ratio_w == crop_ratio.w):
			logger.debug("crop_ratios are not changed.")
			return crop_ratio
		config_doc.update({image_area_key: {'h_crop_r': crop_ratio.h, 'w_crop_r': crop_ratio.w}})
		# image_area_config['h_crop_r'] = crop_ratio
		# image_area_config['w_crop_r'] = crop_ratio_w
		try:
			toml_file.write(config_doc)
			logger.info("Config is saved for crop_ratio as [%s] into file: '%s'", image_area_config, config_filename)
		except IOError as e:
			logger.error("Failed to write config to file: %s", e)
			exit(1)
		return crop_ratio"""


def get_config(filepath: Path, config_dir='Data') -> tuple[TOMLFile, TOMLDocument]: 
	config_path = filepath.parent / config_dir
	config_path.mkdir(exist_ok=True)
	config_filename = config_path / (filepath.stem + '.toml')
	return (f:=TOMLFile(config_filename)), f.read()

def save_config(toml_file: TOMLFile, config_doc: TOMLDocument, config_filename: Path):
	try:
		toml_file.write(config_doc)
		logger.info("Config is saved into file: '%s'", config_filename)
	except IOError as e:
		logger.error("Failed to write config to file: %s", e)
		exit(1)

def get_min_distance(cont: ndarray) -> int: # Sequence[Sequence[int]]) -> int:
	dist_list = []
	for i in range(4):
		p = cont[i, 0] # - cont[i, 0, 0]
		q = cont[(i + 1) % 4, 0] # - cont[i, 0, 0]
		d = floor(dist(p, q))
		if not d:
			return 0
		dist_list.append(d)
		# if d == 0: return 0
		# dw = cont[(i + 1) % 4, 0, 1] - cont[i, 0, 1]
		# dhdw = dh^2 + dw^2
	return min(dist_list)

if __name__ == '__main__':
	config_filename = __file__.rsplit('.')[0] + '.toml'
	try:
		with open(config_filename,'rb') as file:
			config = load_toml_file(file)
		IMAGE_PATH_DICT = config["image-path"]
		dirname = IMAGE_PATH_DICT['dir']
	except FileNotFoundError:
		print(f"Config file {config_filename} does not exist!")
		exit(1)
	except KeyError:
		print(f"Image dir not specified in {config_filename} as 'dir=<dirname>' in section '[image-path]'!")
		exit(1)

	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-f', "--filespec", help="file to process")
	parser.add_option("-a", "--max_aspect_ratio", type="float", default=10.0, help="max aspect ratio")
	parser.add_option("-t", "--threshold", type="float", default=0.5, help="threshold")
	parser.add_option("-v", "--vertical_crop_ratio", type="float", default=1.0, help="vertical crop ratio")
	parser.add_option("-m", "--manual-mask", default=False, action="store_true", help="manually crop mask")
	# parser.add_option("-h", "--help", action="store_true", help="show this help message and exit")
	opts, args = parser.parse_args() # getopt(argv[1:], "ha:t:", ["help", "max_aspect_ratio=", "threshold="]) 
	HELP_OPTION = "-h"
	HELP_TEXT = "Rectangle detector. Needs filespec. Options:: -h: help, -a<float>: max_axpect_ratio(default=10)"
	if not len(args):
		print(HELP_TEXT)
		exit(1)
	filename = args[0]
	if len(args) > 1:
		print(f"Only the first file '{filename}' is specified.")
	img_path = Path(filename)
	if not img_path.exists() and 'dir' in IMAGE_PATH_DICT :
		img_path = Path(IMAGE_PATH_DICT['dir']) / img_path
		if not img_path.exists():
			print(f"Image file {img_path} does not exist!")
			exit(1)
	opt_threshold_ratio = opts.threshold or 0.5
	opt_aspect_ratio = opts.max_aspect_ratio or 10.0
	vertical_crop_ratio = opts.vertical_crop_ratio or 1.0

	IMAGE_AREA_DICT = config.get("image-area") or {}
	image_area = IMAGE_AREA_DICT.get(img_path.stem) or (0, 0)
	IMAGE_ASPECT_DICT = config.get("image-aspect") or {}
	if opt_aspect_ratio is None:
		image_aspect_dict = IMAGE_ASPECT_DICT.get(img_path.stem) 
		param_aspect = image_aspect_dict.get('ratio') if image_aspect_dict else 10.0
	else:
		param_aspect = opt_aspect_ratio
	logger.info("max_aspect_ratio: %f", param_aspect)
	logger.info("threshold_ratio: %f", opt_threshold_ratio)
	main(img_path, threshold_ratio=opt_threshold_ratio, max_rect_aspect=param_aspect, vertical_crop_ratio=vertical_crop_ratio, manual_mask=opts.manual_mask)
# image_area=image_area, 
	# if len(argv) < 2: print("Rectangle detector. Needs filespec.") exit(1) main(argv[1], cutoff=float(argv[2]))