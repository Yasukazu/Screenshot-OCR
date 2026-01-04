from typing import Deque
import numpy as np
import cv2
from pathlib import Path

class OCRFilter:
	THRESHOLD = 235

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

