from itertools import groupby
from pathlib import Path
from collections import deque
from typing import Sequence, Optional
from dataclasses import dataclass
from cv2.gapi import threshold
import numpy as np
import cv2
from image_filter import HeadingAreaParam, FigurePart, XYRange, FromBottomLabelRange, Int4, XYOffset, ImageAreaParamName, ImageDictKey,PaystubAreaParam, BreaktimeAreaParam, ShiftAreaParam, ImageAreaParam

@dataclass
class TaimeeHeadingAreaParam(HeadingAreaParam):
	@classmethod
	def get_label_text_start(cls) -> str:
		return 'この店'

	@classmethod
	def from_image(cls, image: np.ndarray, offset: int = 0, figure_parts: dict[
	FigurePart, XYRange|FromBottomLabelRange] = {}, image_check=False, label_check=True) -> "TaimeeHeadingAreaParam":
		xy_offset = cls.check_image(image, figure_parts=figure_parts, image_check=image_check, label_check=label_check)
		return cls(height=-xy_offset.y, x_offset=xy_offset.x)

	def as_slice_param(self) -> Sequence[Int4]:
		return ((0, self.height, self.x_offset, -1),)

	@classmethod
	def scan_image_range_x(cls, image: np.ndarray)-> range:
		'''rt = stop = -1
		for i, e in enumerate(np.argmax(image[:, :],axis=0)):
			if start == -1:
				if e == 0: # black
					start = i
			else:
				if e != 0: # white
					stop = i
					break'''
			
		x = -1
		black_found = False
		for x in range(image.shape[1]):
			if np.any(image[:, x] == 0):
				black_found = True
				break
		if not black_found:
			raise ValueError("No black found in scan area!")
		x0 = x
		white_found = False
		for x in range(x0, image.shape[1]):
			if np.all(image[:, x] == 255):
				white_found = True
				break
		if not white_found:
			raise ValueError("No white found in scan area!")		
		return range(x0, x)

	@classmethod
	def scan_image_range_y(cls, scan_area: np.ndarray)-> range:
		## scan horizontal lines to find the vertical range of the shape
		y = -1
		black_found = False
		for y in range(scan_area.shape[0]):
			if np.any(scan_area[y, :] == 0):
				black_found = True
				break
		if not black_found:
			raise ValueError("No black found in scan area(2)!")
		y0 = y
		white_found = False
		for y in range(y0, scan_area.shape[0]):
			if np.all(scan_area[y, :] == 255):
				white_found = True
				break
		if not white_found:
			raise ValueError("No white found in scan area(2)!")
		return range(y0, y)

	@classmethod
	def image_shape_check_circle(cls, shape_area: np.ndarray, diff_ratio:float=0.1) -> None:
		canvas = np.full(shape_area.shape[:2], 255, np.uint8)
		cv2.circle(canvas, (shape_area.shape[1]//2, shape_area.shape[0]//2 - 1), shape_area.shape[1]//2, 0, -1)
		# compare shape_area with circle edges(left ang right)
		abs_diff_list = []
		for line in range(shape_area.shape[0]):
			v = shape_area[line, :]
			black_pos = np.where(v == 0)
			if black_pos[0].size > 1:
				l_diff = shape_area[line, black_pos[0][0]] - canvas[line, black_pos[0][0]]
				r_diff = shape_area[line, black_pos[0][-1]] - canvas[line, black_pos[0][-1]]
				abs_diff_list.append(abs(l_diff)+ abs(r_diff))
		abs_diff_sum = sum(abs_diff_list)
		abs_diff_sum_ratio = abs_diff_sum / shape_area.size
		if abs_diff_sum_ratio > diff_ratio:
			raise ValueError("Detected avatar shape is too different from circle!")

	@classmethod
	def check_image(cls, image: np.ndarray, image_check=False, figure_parts: dict[
	FigurePart, XYRange|FromBottomLabelRange] = {},
	avatar_shape_check=False, label_check=True) -> XYOffset:
		'''
		avatar_area | label_area: (y_range, x_range), (from_bottom, from_left)]
		If avatar_area or label_area was/were given as empty list, it/they are fulfilled with found ones.
		returns (x_offset, y_trim) for heading_area should offset x and trim height: y_trim is a negative value, so, adding it to image height'''
		# check if avatar_area is not None
		'''if avatar_area:
			if len(avatar_area) != 2:
				raise ValueError("avatar_area must be a list of 2 elements")
			if not (XYOffset not in avatar_area or XYRange not in avatar_area):
				raise ValueError("avatar area must contain FromBottomFromLeft and XYRange")
			for item in avatar_area:
				if isinstance(item, XYOffset):
					avatar_from_bottom_from_left = item
				else:
					avatar_x_y_range = item'''
		# check if avatar circle at the left side of the area between the borders(1st and 2nd)
		## scan vertical lines to find the horizontal range of the shape (expetcing as a circle)
		if FigurePart.AVATAR not in figure_parts:
			x_range = cls.scan_image_range_x(image)
			y_range = cls.scan_image_range_y(image[:, x_range.start:x_range.stop])
			figure_parts[FigurePart.AVATAR] = XYRange(y=y_range, x=x_range)

		if avatar_shape_check:
			# check black pixel in the shape area
			y_range = figure_parts[FigurePart.AVATAR].y
			x_range = figure_parts[FigurePart.AVATAR].x
			shape_area = image[y_range.start:y_range.stop, x_range.start:x_range.stop] #scan_area[v_range[0]:v_range[1], h_range[0]:h_range[1]]
			cls.image_shape_check_circle(shape_area)

		# shape_area_copy_as_white = np.full(shape_area.shape, 255, np.uint8)
		# cv2.circle(shape_area_copy_as_white, (shape_area.shape[1]//2, shape_area.shape[0]//2), shape_area.shape[1]//2, 0, -1)
		# diff_image = cv2.bitwise_xor(canvas, shape_area_copy_as_white)
		### draw a circle on virtual_circle_area
		'''SUBPLOT_SIZE = 3
		fig, ax = plt.subplots(SUBPLOT_SIZE, 1)#, figsize=(10, 4*SUBPLOT_SIZE))
		for r in range(SUBPLOT_SIZE):
			ax[r].invert_yaxis()
			ax[r].xaxis.tick_top()
			ax[r].set_title(f"Row {r+1}")
		ax[0].imshow(canvas, cmap='gray')
		ax[1].imshow(shape_area, cmap='gray')
		ax[2].imshow(abs_diff_list, cmap='gray')
		plt.show()
		diff_white_count = np.count_nonzero(diff_image != 0)
		diff_white_percentage = diff_white_count * 100 / diff_image.size
		if diff_white_percentage > 5:
			raise ValueError("Detected avatar circle area white diff is too large!")
		shape_area_black_count = np.count_nonzero(shape_area == 0)
		if abs_diff_sum / shape_area_black_count > 0.1:
			raise ValueError("Detected avatar circle area black diff is too large!")
		'''
		if FigurePart.LABEL not in figure_parts:
			# scan label-like shape from bottom, right side of the avatar area
			start_x = figure_parts[FigurePart.AVATAR].x.stop
			scan_area = image[:, start_x:]
			if image_check:
				cv2.imshow("scan_area", scan_area)
				cv2.waitKey(0)
			scan_h, scan_w = scan_area.shape[:2]
			label_w_min = scan_w // 4
			def get_run_length(line: Sequence[int]):
				for n, (k, g) in enumerate(groupby(line)):
					if n == 0:
						_g = g
					elif n == 1 and k == 0 and (w:=len(list(g))) >= label_w_min:
						return _g, w

			label_bottom_line = None
			for y in range(scan_h - 1, 0, -1):
				if (g_w := get_run_length(scan_area[y, :].tolist())):
					label_bottom_line = g_w[1]
					break
			if not label_bottom_line:
				raise ValueError("No label-like shape's bottom line found in heading bottom area!")
			label_bottom_x_offset = len(list(g_w[0]))
			# scan_area[0, :] = 255
			y2 = -1
			bg_found = False
			x = figure_parts[FigurePart.AVATAR].x.stop
			for y2 in range(y, 0, -1):
				if np.all(scan_area[y2, x:] == 255):
					bg_found = True
					break
			if not bg_found:
				raise ValueError("No b.g. above label-like shape!")
			from_bottom_label_range = FromBottomLabelRange(y2 - scan_area.shape[0], range(start_x + label_bottom_x_offset, start_x + label_bottom_x_offset + label_bottom_line))
			if label_check:
				from tesseract_ocr import TesseractOCR, Output
				ocr = TesseractOCR()
				ocr_area = image[image.shape[0] + from_bottom_label_range.from_bottom:, start_x:]
				ocr_result = ocr.exec(ocr_area, output_type=Output.DATAFRAME, psm=7)
				label_text = ''.join(list(ocr_result[ocr_result['conf'] > 50]['text']))
				if not label_text.startswith(
				lts:=cls.get_label_text_start()):
					raise ValueError("Improper text start: " + lts)
			figure_parts[FigurePart.LABEL] = from_bottom_label_range
		return XYOffset(x=figure_parts[FigurePart.AVATAR].x.stop, y=figure_parts[FigurePart.LABEL].from_bottom)

from typing import Deque
class OCRFilter:

	@classmethod
	def get_borders(cls, image: np.ndarray | Path | str, thresh=237, min_bunch=1, max_bunch=10):
		from image_filter import get_horizontal_border_bunches, ImageAreaParamName, _plot
		image = image if isinstance(image, np.ndarray) else cv2.imread(str(image))
		# if not image: raise ValueError("Failed to load image")
		bin_image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
		border_offset_list: Deque[tuple[int, int]] = deque()
		# find borders as bunches
		list(get_horizontal_border_bunches(bin_image, min_bunch=min_bunch, max_bunch=max_bunch,
		 offset_list=border_offset_list))

		margin_range = border_offset_list.popleft()
		y_margin = margin_range[1]
		# bin_image = bin_image[y_margin:, :]
		border_offset_array = np.array(border_offset_list)
		border_offset_array -= y_margin 
		border_offset_ranges = [range(t, p) for t, p in border_offset_array.tolist()]
		check_image = bin_image[y_margin:].copy()
		check_image[:, :len(border_offset_ranges) + 1] = 255
		for n, border_range in enumerate(border_offset_ranges):
			check_image[border_range.start:border_range.stop, n] = 0
		# _plot([bin_image, check_image])
		return y_margin, border_offset_ranges, bin_image

class TaimeeFilter(OCRFilter):
	BORDERS_MIN = 3
	BORDERS_MAX = 4
	LABEL_TEXT_START = 'この店'

	def __init__(self, image: np.ndarray | Path | str, param_dict: dict[ImageAreaParamName, Sequence[int]] = {}, show_check=False):
		from image_filter import get_horizontal_border_bunches, ImageAreaParamName
		self.image = image if isinstance(image, np.ndarray) else cv2.imread(str(image))
		if self.image is None:
			raise ValueError("Failed to load image")
		self.params = param_dict
		bin_image = cv2.threshold(self.image, self.THRESHOLD, 255, cv2.THRESH_BINARY)[1]

		# find borders as bunches
		border_offset_list: deque[tuple[int, int]] = deque()
		# border_offset_list: deque[BorderOffset] = deque()

		# _border_offset_list: list[BorderOffset] = []
		'''with-margin / without-margin
			margin
		 0 HL
			header
		 1 HL
			shift
		 2 HL
			breaktime
		 3 HL
		'''
		# last_offset = 0
		n = -1
		for n, (b, o) in enumerate(get_horizontal_border_bunches(bin_image, min_bunch=4, offset_list=border_offset_list)):
			# if n == 0: y_margin = b.elems[-1] + 1
					# last_b_end = b.elems[-1]
				# elif n == 1: horizontal_border_offset_list.append(BunchOffset(b, b.elems[0] - 1))
			if n == 3:
				break
				# last_offset += b.elems[-1] + 1


		margin_area = border_offset_list.popleft()
		y_margin = border_offset_list[0][0]
		bin_image = bin_image[y_margin:, :]
		border_offset_array = np.array(border_offset_list)
		del(border_offset_list)
		border_offset_array -= y_margin # border_offsets[0][0]
		border_offset_ranges = [range(t, p) for t, p in border_offset_array.tolist()]
		'''if __debug__:
			canvas = bin_image.copy()
			canvas[:, 0:4] = 255
			for n, rg in enumerate(border_offset_ranges):
				canvas[rg.start:rg.stop, n] = 0
			_plot([canvas])'''
		self.y_margin = y_margin

		# self.y_origin = y_origin = border_offsets[0][1]
		heading_area_figure_parts = {}
		try:
			heading_area_param = TaimeeHeadingAreaParam(*param_dict[ImageAreaParamName.heading])
		except (KeyError, TypeError):
			heading_area_param = TaimeeHeadingAreaParam.from_image(bin_image[:border_offset_ranges[0].stop, :], figure_parts=heading_area_figure_parts)

		if show_check:
			show_image = self.image[self.y_margin + heading_area_param.y_offset:self.y_margin + border_offset_ranges[0].stop + heading_area_param.height, heading_area_param.x_offset:]
			do_show_check("heading_area", heading_area_param, show_image)
		self.area_param_dict: dict[ImageAreaParamName, ImageAreaParam] = {ImageAreaParamName.heading: heading_area_param}
		# get shift area
		try:
			area_param = ShiftAreaParam(*param_dict[ImageAreaParamName.shift])
		except (KeyError, TypeError):
			area_range = border_offset_ranges[1] # list[0].elems[-1] + 1
			# y_origin = border_offset_list[1].elems[0]
			# scan_image = bin_image[area_range.start:area_range.stop, :]
			area_param = ShiftAreaParam.from_image(bin_image, offset_range=area_range, image_check=show_check)
		if show_check:
			area_image = bin_image[area_param.y_offset:area_param.y_offset + area_param.height, :]
			do_show_check("shift_area", area_param, area_image)
		self.area_param_dict[ImageAreaParamName.shift] = area_param
		# get breaktime area
		try:
			area_param = BreaktimeAreaParam(*param_dict[ImageAreaParamName.breaktime])
		except (KeyError, TypeError):
			area_range = border_offset_ranges[2]
			# y_origin = border_offset_list[2].elems[0]
			area_param = BreaktimeAreaParam(y_offset=area_range.start, height=area_range.stop - area_range.start) # y_origin - area_top)
		if show_check:
			area_image = bin_image[area_param.y_offset:area_param.y_offset + area_param.height, :]
			do_show_check("breaktime area", area_param, area_image)
		self.area_param_dict[ImageAreaParamName.breaktime] = area_param
		# get paystub area
		try:
			area_param = PaystubAreaParam(*param_dict[ImageAreaParamName.paystub])
		except (KeyError, TypeError):
			# area_range = border_offset_ranges[3] # elems[-1] + 1
			area_param = PaystubAreaParam(y_offset=border_offset_ranges[-1].stop)
		if show_check:
			area_image = bin_image[area_param.y_offset:, :]
			do_show_check("paystub area", area_param, area_image)
		self.area_param_dict[ImageAreaParamName.paystub] = area_param
		if show_check:
			cv2.destroyAllWindows()
		
	def extract_heading(self, params: dict[ImageDictKey, tuple[int, int]] | None = None, seek_button_shape: bool = False, button_text: Optional[list[str]] = None) -> ImageAreaParam|tuple[ImageAreaParam, str]:
		'''Return: (ypos, height, x_start)
		Use with self.y_offset like image[self.y_offset: self.y_offset + height, x_start:]'''
		if len(self.distant_array) == 0:
			raise ValueError("No nearby borders found")
		if self.y_origin >= self.distant_array[0]:
			raise ValueError("Y offset is not less than non-nearby array first element")
		heading_area = bin_image[self.y_origin:self.distant_array[0] + self.y_origin, :].copy()
		for y in self.horizontal_lines:
			if y >= heading_area.shape[0]:
				break
			heading_area[y, :] = 255
		xpos = self.get_heading_avatar_end_xpos(heading_area, remove_borders=False)
		# scan button like shape from bottom
		if seek_button_shape or (button_text is not None):
			heading_h, heading_w = heading_area.shape[:2]
			button_w_min = heading_w // 3
			def get_line(line: Sequence[int]):
				for (k, g) in groupby(line):
					if k == 0 and (w:=len(list(g))) >= button_w_min:
						return w
			def get_hline(y: int):
				for n, (k, g) in enumerate(groupby(heading_area[y, :].tolist())):
					if n == 3 and k == 0 and (w:=len(list(g))) >= button_w_min:
						return w
			button_bottom_line = None
			for y in range(heading_h - 1, 0, -1):
				if (w:=get_line(heading_area[y, :].tolist())):
					button_bottom_line = w
					break
			if not button_bottom_line:
				raise ValueError("No button found in heading bottom area!")
			heading_area[0, :] = 255
			y2 = -1
			bg_found = False
			for y2 in range(y, 0, -1):
				if np.all(heading_area[y2, xpos:] == 255):
					bg_found = True
					break
			if not bg_found:
				raise ValueError("No bg above button shape!")
			button_area = heading_area[y2:y, xpos:]
			cv2.imshow("Button area", button_area)
			cv2.waitKey(0)
			# find contours of the button
			contours, _ = cv2.findContours(button_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			contour = contours[0]
			x, y, w, h = cv2.boundingRect(contour)
			button_area_copy = button_area.copy()
			cv2.rectangle(button_area_copy, (x, y), (x + w, y + h), 0, 2)
			cv2.imshow("Button area and its bounding rect", button_area_copy)
			cv2.waitKey(0)
			print(f"Bottom Shape's bounding rectangle size is:: height: {h}, width: {w} ")
			h_area_copy = heading_area.copy()
			cv2.rectangle(h_area_copy, (xpos + x, y2), (xpos + x2, y), (0, 0, 0), 2)
			cv2.imshow("Heading area bottom shape", h_area_copy)	
			cv2.waitKey(0)
			# button_text: str | None = None
			if button_text is not None:
				from tesseract_ocr import TesseractOCR, Output
				ocr = TesseractOCR()
				ocr_result = ocr.exec(heading_area[y2:y, xpos:], output_type=Output.DATAFRAME, psm=7)
				button_text += list(ocr_result[ocr_result['conf'] > 50]['text'])
				# print(f"OCR Result text: {ocr_text}")

			assert np.any(heading_area[y2 + 1, :] != 255)
			button_top_line = get_line(heading_area[y2 + 1,:].tolist()) # length
			if abs(button_top_line - button_bottom_line) > 10:
				raise ValueError(f"Button shape is not top-bottom symmetrical! top: {button_top_line}, bottom: {button_bottom_line}")	
			'''cv2.imshow("Heading area bottom shape", heading_area[y2+1:y+1, xpos:])	
			cv2.waitKey(0)'''
			# try to find button width
			button_band = heading_area[y2+1:y+1, xpos:]
			shape_w = button_band.shape[1]
			black_found = False
			for x in range(shape_w):
				if np.any(button_band[:, x] != 255):
					black_found = True
					break
			if not black_found:
				raise ValueError("No valid shape found (3)")
			bg_found = False
			for x2 in range(x, shape_w):
				if np.all(button_band[:, x2] == 255):
					bg_found = True
					break
			if not bg_found:
				raise ValueError("No valid shape found (4)")
			print(f"Button Shape's rectangle size is:: height: {y - y2}, width: {x2 - x} ")
			print(f"and it's position is:: from heading area top: {y2}, from left: {xpos + x}")
			assert (y + 2 ) > (y2 - 1)
			assert (x2 + 1) > (x - 1)
			button_rect = button_band[:, x:x2]
			# cv2.imshow("Heading shape area ", shape_band)	
			cv2.imshow("Heading area bottom shape rect", button_rect)	
			cv2.waitKey(0)


		new_params = ImageAreaParam(0, y2, xpos)#, -1)
		if params is not None:
			params[ImageDictKey.heading] = new_params
		self.params = params
		return new_params

		

		
	@classmethod
	def get_heading_avatar_end_xpos(cls, image: np.ndarray, min_width: int = 8, 		borders: list[int] | None =None, remove_borders: bool = False)->int:
		height, width = image.shape[:2]

		# Remove horizontal borders from the image to simplify processing
		if remove_borders:
			image = image.copy()
			if borders is None:
				for y in (find_horizontal_borders(image)):
					image[y, :] = 255
			else:
				for y in borders:
					image[y, :] = 255

		px = -1
		for px in range(width):
			v_line = image[:, px]
			if np.any(v_line == 0):
				break
		if px == width:
			raise ValueError("Not enough valid width(%d) for the heading!" % width)
		# 2 pass the shape
		x = -1
		for x in range(px + 1, width):
			v_line = image[:, x]
			if np.all(v_line == 255):
				break
		if x < min_width:
			raise ValueError("Not enough valid width(%d) for the heading!" % x)
		return x

if __name__ == '__main__':
	from sys import argv
	image_path = argv[1]
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	y_margin, borders, bin_image = OCRFilter.get_borders(image)
	border_end_list = [b.stop for b in borders]
	first_border_end = border_end_list[0]
	border_ratio_list = [b.stop / first_border_end for b in borders[1:]]
	print(','.join([f"{b:.1f}" for b in border_ratio_list]))