from tomllib import load as load_toml_file
from math import sqrt, floor, dist
from typing import Sequence
from sys import argv, exit, stdout
import cv2 
import numpy as np 
from numpy import ndarray
from logging import getLogger, INFO, StreamHandler, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)
stdout_handler = StreamHandler(stdout)
stdout_handler.setLevel(DEBUG)
logger.addHandler(stdout_handler)

config_filename = __file__.rsplit('.', 1)[0] + '.toml'
with open(config_filename,'rb') as file:
	config = load_toml_file(file)
IMAGE_AREA_DICT = config["image-area"]
from sys import exit
from pathlib import Path
def main(filename: str | Path, cutoff: int=5, BGR='B'):
	if cutoff < 1:
		logger.error("cutoff must be greater than 0!")
		exit(1)
	logger.info("main started.")
	# Load the image 
	image = cv2.imread(str(filename)) # 'path/to/your/image.jpg')
	if not image:
		logger.error("Failed to load image: %s", filename)
		exit(1)
	image_h, image_w, _ = image.shape 
	assert image_h and image_w
	assert image_h > cutoff and image_w > cutoff
	min_image_size = min([image_h, image_w])
	min_radius = min_image_size * cutoff // 100 // 2
	logger.debug("image_h: %d, image_w: %d, min_radius: %d", image_h, image_w, min_radius)
	# Parameters window
	def empty(x):
		pass

	PARAMS = 'Parameters'
	cv2.namedWindow(PARAMS)
	cv2.resizeWindow(PARAMS, width=800, height=800)
	cv2.createTrackbar('k_size_set', PARAMS, 1, 10, empty) #3
	cv2.createTrackbar('canny_1st', PARAMS, 90, 500, empty) #80
	cv2.createTrackbar('canny_2nd', PARAMS, 60, 500, empty) #120
	cv2.createTrackbar('minDist_set', PARAMS, 100, 200, empty)
	cv2.createTrackbar('param1_set', PARAMS, 100, 300, empty) #100
	cv2.createTrackbar('param2_set', PARAMS, 30, 300, empty) #30
	cv2.createTrackbar('minRadius_set', PARAMS, min_radius, min_image_size, empty) #250
	cv2.createTrackbar('maxRadius_set', PARAMS, 0, min_image_size, empty ) #500


	while True:
		src_f = image.astype(np.float64).copy()
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
		cv2.imshow('Luminosity result', luminosity_result)
		cv2.waitKey(0) 
		# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
		
		# Apply Gaussian blur to reduce noise and improve edge detection 
		blurred = cv2.GaussianBlur(luminosity_result, (7, 7), 3) # medianBlur
		cv2.imshow('Blur', blurred)
		cv2.waitKey(0) 
		kernel = cv2.getTrackbarPos("k_size_set", "Parameters")
		kernel = (kernel * 2) + 1
		img_blur = cv2.GaussianBlur(img_gray, (kernel, kernel), None)
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
		# Binarize
		ret, binarized = cv2.threshold(blurred, 200, 255,cv2.THRESH_BINARY)
		# binarized = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
		cv2.imshow('Binarized', binarized)
		cv2.waitKey(0)
		"""
		# Canny edge detection 
		image_edges = cv2.Canny(blurred, 50, 150) 
		cv2.imshow('Detected edges', image_edges)
		cv2.waitKey(0) 
		# Find contours in the edged image 
		contours, _ = cv2.findContours(image_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
		if not len(contours):
			raise ValueError("No contours detected!")
		logger.info("%s contours detected.", len(contours))
		#limit = 4 # len(contours) // 32
		#limited_cont = []
		sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True) #[:limit] # (limit // 2):limit] 
		# if len(ct) == 4: limited_cont.append(ct) if len(limited_cont) >= limit: break
		# logger.info("%s contours limited.", len(limited_cont))
		# image_cutoff = floor(image_min * cutoff) # / 16) or 4
		# Loop over the contours 
		limit = 16
		added = 0
		for contour in sorted_contours: 
			# Approximate the contour to a polygon 
			epsilon = 0.02 * cv2.arcLength(contour, True) 
			approx_cont = cv2.approxPolyDP(contour, epsilon, True) 
			# Check if the approximated contour has 4 points (rectangle) 
			# if len(approx_cont) == 4:
				# min_dist = get_min_distance(approx_cont)
				# if min_dist > image_cutoff: 
					# Draw the rectangle on the original image 
			cv2.drawContours(image, [approx_cont], -1, (0xff, 0, 0), 2) 
			cv2.imshow('Detected Rectangles', image) 
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
	from getopt import getopt
	opts, args = getopt(argv[1:], "hi:", ["help", "min_diameter_percentage="]) 
	HELP_OPTION = "-h"
	HELP_TEXT = "Circle detector. Needs filespec. Options:: -h: help, -i<int>: min_diameter_percentage>"
	if not len(args):
		print(HELP_TEXT)
		exit(1)
	img_path = Path(args[0])
	if not img_path.exists():
		print(f"Image file {img_path} does not exist!")
		exit(1)
	min_diameter_percentage = 1
	for opt, arg in opts:
		match opt:
			case "-h":
				print(HELP_TEXT)
				exit(0)
			case "-i":
				min_diameter_percentage = int(arg)	
			case _:
				logger.error("Invalid option: %s", opt)
				exit(1)
	main(img_path, cutoff=min_diameter_percentage)