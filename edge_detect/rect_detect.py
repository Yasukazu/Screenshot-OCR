from tomllib import load as load_toml_file
from math import sqrt, floor, dist
from typing import Sequence
from sys import argv, exit, stdout

import cv2 
import numpy as np 
from numpy import ndarray
import matplotlib.pyplot as plt

from logging import getLogger, INFO, StreamHandler, DEBUG
logger = getLogger(__name__)
logger.setLevel(INFO)
stdout_handler = StreamHandler(stdout)
stdout_handler.setLevel(INFO)
logger.addHandler(stdout_handler)

from pathlib import Path
from typing import NamedTuple

class Rect(NamedTuple):
	x: int
	y: int
	w: int
	h: int

# def get_aspect_ratio(cont: ndarray) -> float: rect = cv2.minAreaRect(cont)

def main(filename: str | Path, threshold_ratio=0.5, BGR='B', image_area=(0, 0), max_rect_aspect=10.0, vertical_crop_ratio=1.0):
	logger.debug("%f max_rect_aspect", max_rect_aspect)
	logger.info("%f threshold_ratio:level %f", threshold_ratio, threshold_ratio * 255)
	logger.info("%f vartical_crop_ratio", vertical_crop_ratio)
	# Load the image 
	image = cv2.imread(str(filename)) # 'path/to/your/image.jpg') 
	if image is None:
		logger.error("Failed to load image: %s", filename)
		exit(1)
	image_h, image_w, _ = image.shape 
	logger.debug("image_h: %d, image_w: %d", image_h, image_w)
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
	cv2.waitKey()

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

	""" # Sobel filter
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
	"""
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
			cv2.drawContours(image, [contour], -1, (0xff, 0, 0), 2) 
			cv2.imshow('Detected bounding rectangles', image) 
			cv2.waitKey(20) 
			added += 1
				# if added >= limit: break
	
	# Display the result 
	cv2.waitKey(0) 

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
	main(img_path, threshold_ratio=opt_threshold_ratio, image_area=image_area, max_rect_aspect=param_aspect, vertical_crop_ratio=vertical_crop_ratio)

	# if len(argv) < 2: print("Rectangle detector. Needs filespec.") exit(1) main(argv[1], cutoff=float(argv[2]))