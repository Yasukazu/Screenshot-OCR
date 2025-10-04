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
# IMAGE_AREA_DICT = config["image-area"]
IMAGE_PATH_DICT = config["image-path"]
from sys import exit
from pathlib import Path
def empty(x):
	pass
def main(filename: str | Path, cutoff: int=5, BGR='B'):
	if cutoff < 1:
		logger.error("cutoff must be greater than 0!")
		exit(1)
	logger.info("main started.")
	# Load the image 
	image = cv2.imread(str(filename)) # 'path/to/your/image.jpg')
	if image is None:
		logger.error("Failed to load image: %s", filename)
		exit(1)
	image_h, image_w, _ = image.shape 
	assert image_h and image_w
	assert image_h > cutoff and image_w > cutoff
	min_image_size = min([image_h, image_w])
	min_radius = min_image_size * cutoff // 100 // 2
	logger.debug("image_h: %d, image_w: %d, min_radius: %d", image_h, image_w, min_radius)
	# Parameters window


	PARAMS = 'Parameters'
	cv2.namedWindow(PARAMS)
	cv2.resizeWindow(PARAMS, width=800, height=800)
	cv2.createTrackbar('k_size_set', PARAMS, 1, 10, empty) #3
	cv2.createTrackbar('canny_1st', PARAMS, 90, 500, empty) #80
	cv2.createTrackbar('canny_2nd', PARAMS, 60, 500, empty) #120
	cv2.createTrackbar('minDist_set', PARAMS, 100, 200, empty)
	cv2.createTrackbar('param1_set', PARAMS, 100, 300, empty) #100
	cv2.createTrackbar('param2_set', PARAMS, 30, 300, empty) #30
	cv2.createTrackbar('minRadius_set', PARAMS, min_radius, min_image_size // 2, empty) #250
	cv2.createTrackbar('maxRadius_set', PARAMS, min_image_size // 2, min_image_size // 2, empty ) #500

	# Convert to grayscale 
	img_gray = np.zeros((image_h, image_w), np.uint8)
	# ルミナンス法（Luminosity Method）
	bgr_pos = 'BGR'.index(BGR)
	for i in range(image_h):
		for j in range(image_w):
			#blue = src_f[i, j, 0]
			#green = src_f[i, j, 1]
			#red = src_f[i, j, 2]
			img_gray[i, j] = image[i, j, bgr_pos] 

	while True:
		img_dst = img_gray.copy() # astype(np.float64).
		# cv2.imshow('dst image', img_dst) cv2.waitKey(0) 
# 0.02126*red + 0.5152*green + 0.722*blue
		# cv2.imshow('Luminosity result', img_gray)
		# cv2.waitKey(0) 
		# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
		
		# Apply Gaussian blur to reduce noise and improve edge detection 
		# blurred = cv2.GaussianBlur(img_gray, (7, 7), 3) # medianBlur
		# cv2.imshow('Blur', blurred)
		# cv2.waitKey(0) 
		kernel = cv2.getTrackbarPos("k_size_set", "Parameters")
		kernel = (kernel * 2) + 1
		img_blur = cv2.GaussianBlur(img_dst, (kernel, kernel), None)
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
		thres1_val = cv2.getTrackbarPos('canny_1st', 'Parameters')
		thres2_val = cv2.getTrackbarPos('canny_2nd', 'Parameters')
		img_edges = cv2.Canny(img_blur, threshold1=thres1_val, threshold2=thres2_val)
		
		# image_edges = cv2.Canny(blurred, 50, 150) 
		# cv2.imshow('Detected edges', img_edges)
		# cv2.waitKey(0) 
		# Find contours in the edged image 
		circles = cv2.HoughCircles(img_edges, cv2.HOUGH_GRADIENT,
						dp=1,
						minDist=cv2.getTrackbarPos('minDist_set', 'Parameters'),
						param1=cv2.getTrackbarPos('param1_set', 'Parameters'),
						param2=cv2.getTrackbarPos('param2_set', 'Parameters'),
						minRadius=cv2.getTrackbarPos('minRadius_set', 'Parameters'),
						maxRadius=cv2.getTrackbarPos('maxRadius_set', 'Parameters'),
						)
	
		try:
			circles = np.uint16(np.around(circles))
			# cv2.imshow('before curcle draw', img_dst) cv2.waitKey(0)
			for n, circle in enumerate(circles[0, :]):
				logger.info("%d. Radius: %d, center: (%d, %d)", n + 1, circle[2], circle[0], circle[1])
				# 円周を描画する
				cv2.circle(img_dst, (circle[0], circle[1]), circle[2], 0, 4) # (0, 165, 255), 5)
				cv2.rectangle(img_dst, (0, 0), (circle[0] + circle[2], circle[1] + circle[2]), 0, 4)
				cv2.imshow('first circle', img_dst)
				cv2.waitKey(0)
				# print('radius')
				# print(circle[2])
				# 中心点を描画する
				# draw rectangle to cut circle
				# print('center')
				# print(circle[0], circle[1])
			# 4. Plotting
			cv2.imshow('result', img_dst)
			cv2.waitKey(0)
			
		except:
			pass

		# qを押すと止まる。
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		cv2.destroyAllWindows()



if __name__ == '__main__':
	from getopt import getopt
	default_min_diameter_percentage = 4
	opts, args = getopt(argv[1:], "hi:", ["help", "min_diameter_percentage="]) 
	HELP_OPTION = "-h"
	HELP_TEXT = f"Circle detector. Needs filespec. Options:: -h: help, -i<int>: min_diameter_percentage(default={default_min_diameter_percentage})"
	if not len(args):
		print(HELP_TEXT)
		exit(1)
	img_path = Path(args[0])
	if not img_path.exists():
		img_path = Path(IMAGE_PATH_DICT['dir']) / img_path
		if not img_path.exists():
			print(f"Image file {img_path} does not exist!")
			exit(1)
	min_diameter_percentage = default_min_diameter_percentage
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