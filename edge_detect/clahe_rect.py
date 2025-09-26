from collections import deque
from dataclasses import dataclass
from logging.config import fileConfig
from logging import getLogger, INFO
import cv2
import numpy as np

fileConfig("logging-debug.conf")
logger = getLogger(__name__)
logger.setLevel(INFO)
from typing import NamedTuple
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
	clh = clahe(g, 4, (8, 8))

	# Adaptive Thresholding to isolate the bed
	img_blur = cv2.blur(clh, (9, 9))
	img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								cv2.THRESH_BINARY, 51, 2)

	contours, hierarchy = cv2.findContours(img_th,
											cv2.RETR_CCOMP,
											cv2.CHAIN_APPROX_TC89_L1)

	if not len(contours):
		raise ValueError("No contours detected!")
	logger.info("%s contours detected.", len(contours))
	# Filter the rectangle by choosing only the big ones
	# and choose the brightest rectangle as the bed
	max_brightness = 0
	BRIGHT_LIST_SIZE = 4
	b_que = deque([], maxlen=BRIGHT_LIST_SIZE)
	canvas = src.copy()
	brightest_rectangle = None
	src_whr = np.prod(src.shape[:2]) // 8
	logger.info("src_whr is set to %d", src_whr)
	cropped_dict = {}
	for cnt in contours:
		x, y, w, h = rect = Rect(*cv2.boundingRect(cnt))
		if (wh:=w*h) > src_whr: # 40000:
			mask = np.zeros(src.shape, np.uint8)
			mask[y:y+h, x:x+w] = cropped = src[y:y+h, x:x+w]
			# brightness = np.sum(mask)
			if wh > max_brightness:
				brightest_rectangle = rect
				b_que.appendleft(rect)
				max_brightness = wh # brightness
				cropped_dict[rect] = cropped
			# cv2.imshow("mask", mask)
	logger.info("%d cropped rects are found.", len(b_que))
	for rect in b_que: 
		cv2.imshow("cropped", cropped_dict[rect])
		cv2.waitKey(0)
	BGR_LIST = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
	for n, rect in enumerate(b_que): # if brightest_rectangle:
		x, y, w, h = rect # brightest_rectangle
		cv2.rectangle(canvas, (x, y), (x+w, y+h), BGR_LIST[n], 2) # (0, 255, 0), 2)
	cv2.imshow("Canvas [KBGR]", canvas)
		# cv2.imwrite("result.jpg", canvas)
	cv2.waitKey(0)

def clahe(img, clip_limit=4, grid_size=(8,8)):
	clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
	return clahe.apply(img)

if __name__ == '__main__':
	from sys import argv
	main(argv[1])
