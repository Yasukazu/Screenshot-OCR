from collections import deque
from typing import Deque, Sequence
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from image_filter import ImageAreaParamName, ImageAreaParam

class OCRFilter:
	THRESHOLD = 235
	def __init__(self, image:np.ndarray, param_dict:dict[ImageAreaParamName, Sequence[int] | ImageAreaParam] = {k:[] for k in ImageAreaParamName}, show_check=False, thresh=THRESHOLD, bin_image:np.ndarray | None = None):
		self.image = image
		self._param_dict = param_dict
		self.show_check = show_check
		self.thresh = thresh
		self.bin_image = bin_image
		dct: dict[ImageAreaParamName, ImageAreaParam] = {}
		for name in ImageAreaParamName:
			try:
				value = self._param_dict[name]
			except KeyError:
				continue
			if isinstance(value, ImageAreaParam):
				param = value
			else:
				param = name.value
				dct[name] = param(*value)
		self._area_param_dict: dict[ImageAreaParamName, ImageAreaParam] = dct
	
	@property
	def y_margin(self):
		return 0

	@property
	def param_dict(self) -> dict[ImageAreaParamName, ImageAreaParam]:
		if self._area_param_dict is not None:
			return self._area_param_dict
		dct: dict[ImageAreaParamName, ImageAreaParam] = {}
		for name in ImageAreaParamName:
			try:
				value = self._param_dict[name]
			except KeyError:
				continue
			if isinstance(value, ImageAreaParam):
				param = value
			else:
				param = name.value
				dct[name] = param(*value)
		self._area_param_dict = dct
		return dct

	@classmethod
	def convert_border_offset_ranges_to_ratio_list(cls, borders: list[range]) -> list[float]:
		ratio_list = []
		base_len = borders[0].stop
		for b in borders[1:]:
			ratio_list.append(b.stop / base_len)
		return ratio_list
	
	@classmethod
	def get_borders(cls, image: np.ndarray | Path | str, thresh=237, min_bunch=1, max_bunch=10, check_image=False) -> tuple[int, list[range], np.ndarray]:
		from image_filter import get_horizontal_border_bunches
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
		if check_image:
			from image_filter import _plot
			canvas = bin_image[y_margin:].copy()
			canvas[:, :len(border_offset_ranges) + 1] = 255
			for n, border_range in enumerate(border_offset_ranges):
				canvas[border_range.start:border_range.stop, n] = 0
			_plot([bin_image, canvas])
		return y_margin, border_offset_ranges, bin_image

