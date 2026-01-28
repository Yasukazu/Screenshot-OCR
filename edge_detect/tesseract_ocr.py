from enum import Enum, StrEnum, auto
from returns.result import safe
from pytesseract import pytesseract, image_to_data, image_to_boxes
from pathlib import Path
import numpy as np
class Output(Enum):
	BYTES = 'bytes'
	DATAFRAME = 'data.frame'
	DICT = 'dict'
	STRING = 'string'

class TesseractLang(StrEnum):
	JPN = auto()
	ENG = auto()

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

	@safe
	def exec(self, image: np.ndarray, langs: set[TesseractLang]=set(TesseractLang), output_type=Output.DICT, data_or_boxes=True, psm:int|None=None):
		if image.size == 0:
			raise ValueError("Image size is 0")
		if psm is not None:
			psm_value = int(psm)
		else:
			psm_value = self.psm_value
		psm_config = f"--psm {psm_value}"
		lang='+'.join(langs)
		return image_to_data(image, lang=lang, output_type=output_type.value, config=' '.join([self.tessdata_dir_config, psm_config])) if data_or_boxes else image_to_boxes(image, lang=lang, output_type=output_type.value, config=' '.join([self.tessdata_dir_config, psm_config]))