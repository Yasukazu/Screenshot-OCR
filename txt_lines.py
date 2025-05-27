from typing import Sequence, Union
from contextlib import closing
from enum import IntEnum
from types import MappingProxyType
import re
from typing import Sequence, Callable, Any
from pprint import pp
from pathlib import Path
from dataclasses import dataclass
import pickle
import sqlite3
from returns.result import safe, Result, Failure, Success
import pandas
from ipdb import set_trace as breakpoint
from dotenv import load_dotenv
load_dotenv()
from PIL import Image, ImageDraw, ImageEnhance
import pyocr
import pyocr.builders
from pyocr.builders import LineBoxBuilder, TextBuilder, DigitLineBoxBuilder, DigitBuilder, LineBox, Box, WordBoxBuilder
from returns.pipeline import is_successful, UnwrapFailedError
import loguru # logging
logger = loguru.logger # logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.remove()
import sys
#logger.add(sys.stderr, level="INFO")
logger.add(sys.stdout, level="WARNING")
logger.add("ERROR.log", level="ERROR")
from app_type import AppType
from tool_pyocr import PathSet, MyOcr, MonthDay
from path_feeder import input_dir_root
class DigitLines:
    def __init__(self, stem: str, direc: Path=input_dir_root, ext='.png', year=0, month=0):
        self.path_set = PathSet(parent=direc, stem=stem, ext=ext)
        self.my_ocr = MyOcr(year=year, month=month)
        self.digit_lines: Sequence[Any]|None = None
    def run_ocr(self) -> Sequence[Any]|None:
        breakpoint()
        result = self.my_ocr.run_ocr(path_set=self.path_set, lang='jpn+eng', builder_class=DigitLineBoxBuilder, layout=7) 
        match result:
            case Success(self.digit_lines):
                return self.digit_lines
            case Failure(_):
                return None #raise ValueError(f"Failed to run OCR and get data!")

class TxtLines:
    app_type = AppType.NUL
    def __init__(self, txt_lines: Sequence[LineBox], img_pathset: PathSet, my_ocr: MyOcr = MyOcr()):
        self.my_ocr = my_ocr
        if not img_pathset.exists():
            raise FileNotFoundError(f"Image path {img_pathset} does not exist!")
        self.txt_lines = txt_lines
        self.img_pathset = img_pathset

class TTxtLines(TxtLines):
    """
    Class to handle text lines from T app..
    """
    app_type = AppType.T

    def __init__(self, txt_lines: Sequence[LineBox], img_pathset: PathSet, my_ocr: MyOcr = MyOcr()):
        super().__init__(txt_lines, img_pathset=img_pathset, my_ocr=my_ocr)

    def get_date(self) -> MonthDay:
        breakpoint()
        for n, txt_line in enumerate(self.txt_lines):
            if txt_line.content.replace(' ', '').startswith('業務開始'):
                break
        if n >= len(self.txt_lines) - 1:
            logger.error("No date found in txt_lines for stem: {}", self.img_pathset.stem)
            raise ValueError(f"No date found in txt_lines for stem: {self.img_pathset.stem}")
        date_position = self.txt_lines[n + 1].position
        date_position = date_position[0] + date_position[1]
        breakpoint()
        img_path = self.img_pathset #.parent / (self.img_pathset.stem + self.img_pathset.ext)
        date_image = Image.open(str(img_path)).crop(date_position)
        breakpoint()
        date_image_dir = self.img_pathset.parent.parent / 'TMP'
        date_image_dir.mkdir(parents=True, exist_ok=True)
        date_image_fullpath = date_image_dir / f'{self.img_pathset.stem}.date.png'
        date_image.save(date_image_fullpath, format='PNG')
        logger.info("Saved date image: {}", date_image_fullpath)
        result = self.my_ocr.run_ocr(path_set=date_image_fullpath, lang='eng+jpn', builder_class=pyocr.builders.TextBuilder, layout=7)
        match result:
            case Success(value):
                no_spc_value = value.replace(' ', '')
                mt = re.match(r"(\d+)月(\d+)日", no_spc_value)
                if mt and len(mt.groups()) == 2:
                    month, day = mt.groups()
                    date = MonthDay(int(month), int(day))
                    return date
            case Failure(_):
                logger.error("No date found in txt_lines for stem: {}", self.img_pathset.stem)
                raise ValueError(f"No date found in txt_lines for stem: {self.img_pathset.stem}")
