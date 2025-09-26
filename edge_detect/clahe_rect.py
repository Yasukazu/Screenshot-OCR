from collections import deque
from typing import NamedTuple
# from dataclasses import dataclass
from logging.config import fileConfig
from logging import getLogger, INFO, DEBUG
import cv2
import numpy as np

fileConfig("logging-debug.conf")
logger = getLogger(__name__)
logger.setLevel(DEBUG)
class Rect(NamedTuple):
	x: int
	y: int
	w: int
	h: int
	# @property def wh(self): return self.w * self.h

def main(filename: str):
	src = cv2.imread(filename)

	# HSV thresholding to get rid of as much background as possible
	hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV)
	lower_blue = np.array([0, 0, 120])
	upper_blue = np.array([180, 38, 255])
	mask = cv2.inRange(hsv, lower_blue, upper_blue)
	result = cv2.bitwise_and(src, src, mask=mask)
	b, g, r = cv2.split(result)
	cv2.imshow('Blue', b)
	cv2.waitKey(0)

	# CLAHE
	clh = clahe(b, 4, (8, 8))
	cv2.imshow('CLAHE', clh)
	cv2.waitKey(0)
	# Blur
	'''img_blur = cv2.blur(clh, (9, 9))
	cv2.imshow('Blur', img_blur)
	cv2.waitKey(0)'''
	# Adaptive Thresholding to isolate the bed
	img_th = cv2.adaptiveThreshold(clh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								cv2.THRESH_BINARY, 51, 2)
	cv2.imshow('Threshhold', img_th)
	cv2.waitKey(0)
	contours, hierarchy = cv2.findContours(img_th,
											cv2.RETR_LIST,
											cv2.CHAIN_APPROX_SIMPLE)#RETR_CCOMP,TC89_L1)

	if not len(contours):
		raise ValueError("No contours detected!")
	logger.info("%s contours detected.", len(contours))
	# Filter contours with enough area
	area_thresh = np.prod(src.shape[:2]) // 64
	logger.info("area_thresh is set to %d", area_thresh)
	contours = list(filter(lambda x: cv2.contourArea(x) > area_thresh, contours))
	logger.info("%s contours remains after filtering.", len(contours))
	# Filter the rectangle by choosing only the big ones
	# and choose the brightest rectangle as the bed
	max_brightness = 0
	BRIGHT_LIST_SIZE = 4
	b_que = deque([], maxlen=BRIGHT_LIST_SIZE)
	brightest_rectangle = None
	src_whr = np.prod(src.shape[:2]) // 8
	logger.info("src_whr is set to %d", src_whr)
	# cropped_dict = {}
	for cnt in contours:
		x, y, w, h = rect = Rect(*cv2.boundingRect(cnt))
		if (wh:=w*h) > src_whr: # 40000:
			# mask = np.zeros(src.shape, np.uint8)
			# mask[y:y+h, x:x+w] = cropped = src[y:y+h, x:x+w]
			brightness = np.sum(src[y:y+h, x:x+w])
			if wh > max_brightness:
				brightest_rectangle = rect
				b_que.appendleft(rect)
				max_brightness = wh # brightness
				# cropped_dict[rect] = cropped
			# cv2.imshow("mask", mask)
	logger.info("%d cropped rects are found.", len(b_que))
	# for rect in b_que: 
	# 	cv2.imshow("Cropped", cropped_dict[rect])
	# 	kbd = cv2.waitKey(0)
	BGR_LIST = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
	canvas = src.copy()
	for n, rect in enumerate(b_que): # if brightest_rectangle:
		x, y, w, h = rect # brightest_rectangle
		cv2.rectangle(canvas, (x, y), (x+w, y+h), BGR_LIST[n], 2) # (0, 255, 0), 2)
	for n, rect in enumerate(b_que):
		ganvas = canvas.copy()
		x, y, w, h = rect # brightest_rectangle
		cv2.rectangle(ganvas, (x, y), (x+w, y+h), BGR_LIST[n], 8) # (0, 255, 0), 2)
		cv2.imshow("Hit 's' to save:", ganvas) # [KBGR:0123]
		# cv2.imwrite("result.jpg", canvas)
		kbd = cv2.waitKey(0)
		if kbd == ord('s'):
			crop_filename = filename.rsplit('.', 1)[0] + '-crop.png'
			logger.debug("'%s' as crop filename.", crop_filename)
			cv2.imwrite(crop_filename, src[y:y+h, x:x+w])

def clahe(img, clip_limit=4, grid_size=(8,8)):
	clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
	return clahe.apply(img)

if __name__ == '__main__':
	from sys import argv
	main(argv[1])
