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
'''['BaseBuilder',
 'Box',
 'DigitBuilder',
 'DigitLineBoxBuilder',
 'HTMLParser',
 'LineBox',
 'LineBoxBuilder',
 'TextBuilder',
 'WordBoxBuilder','''
''' TextBuilder            ：　文字列を認識
    WordBoxBuilder      ：　単語単位で文字認識 + BoundingBox
    LineBoxBuilder       ：　行単位で文字認識 + BoundingBox
    DigitBuilder            ：　数字 / 記号を認識
    DigitLineBoxBuilder ：　数字 / 記号を認識 + BoundingBox'''
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
from tool_pyocr import PathSet, MyOcr
from path_feeder import input_dir_root
class DigitLines:
    def __init__(self, stem: str, direc: Path=input_dir_root, ext='.png', year=0, month=0):
        self.path_set = PathSet(path=direc, stem=stem, ext=ext)
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
    def __init__(self, txt_lines: Sequence[LineBox], img_path: PathSet):
        self.txt_lines = txt_lines
        self.img_path = img_path

class TTxtLines(TxtLines):
    """
    Class to handle text lines from T app..
    """
    app_type = AppType.T

    def __init__(self, txt_lines: Sequence[LineBox], img_path: PathSet):
        super().__init__(txt_lines, img_path)

