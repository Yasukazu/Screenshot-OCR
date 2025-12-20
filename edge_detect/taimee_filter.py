from typing import Sequence
from dataclasses import dataclass
import numpy as np
import cv2
from image_filter import HeadingAreaParam, FigurePart, XYRange, FromBottomLabelRange, Int4, XYOffset

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

