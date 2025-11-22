
from pytesseract import pytesseract, image_to_data, image_to_boxes, Output
from pathlib import Path
import numpy as np

class TesseractOCR:
	def __init__(self, tessdata_dir = "~/.local/share/tessdata/fast", tesseract_cmd = '/usr/bin/tesseract', psm_value = 6):
		self.tessdata_dir = tessdata_dir
		self.tesseract_cmd = tesseract_cmd
		self.conf_min=80
		pytesseract.tesseract_cmd = self.tesseract_cmd
	# with TemporaryDirectory() as tmpdirname: tmp_img_path = '/'.join([tmpdirname, 'test.png']) cv2.imwrite(tmp_img_path, image) text, boxes, tsv = pytesseract.run_and_get_multiple_output(tmp_img_path, extensions=['txt', 'box', 'tsv'])
		self.psm_value = psm_value

	@property
	def psm_config(self):
		return "--psm %d" % self.psm_value

	@property
	def tessdata_dir_config(self):
		tessdata_path = Path(self.tessdata_dir).expanduser()
		return f'--tessdata-dir {tessdata_path}'

	def exec_ocr(self, image: np.ndarray, lang="jpn",output_type=Output.DICT, data_or_boxes=True):
		return image_to_data(image, lang=lang, output_type=output_type, config=' '.join([self.tessdata_dir_config, self.psm_config])) if data_or_boxes else image_to_boxes(image, lang=lang, output_type=output_type, config=' '.join([self.tessdata_dir_config, self.psm_config]))