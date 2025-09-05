from typing import Sequence, Union
from contextlib import closing
from enum import IntEnum
from types import MappingProxyType
import re, sys
from typing import Sequence, Callable, Any
from pprint import pp
from pathlib import Path
from dataclasses import dataclass
import pickle
import sqlite3
from returns.result import safe, Result, Failure, Success
import pandas

from dotenv import load_dotenv
load_dotenv(override=True)
from PIL import Image, ImageDraw, ImageEnhance
import pyocr
import pyocr.builders
from pyocr.builders import LineBoxBuilder, TextBuilder, DigitLineBoxBuilder, DigitBuilder, LineBox, Box, WordBoxBuilder
from returns.pipeline import is_successful, UnwrapFailedError
import logbook

logbook.StreamHandler(sys.stdout,
	format_string='{record.time:%Y-%m-%d %H:%M:%S.%f} {record.level_name} {record.filename}:{record.lineno}: {record.message}').push_application()

logger = logbook.Logger(__file__)
logger.level = logbook.INFO

#import loguru # logging
#logger = loguru.logger # logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#logger.remove()
import sys
#logger.add(sys.stderr, level="INFO")
#logger.add(sys.stdout, level="WARNING")
#logger.add("ERROR.log", level="ERROR")
from app_type import AppType
from path_feeder import input_dir_root
from tool_pyocr import PathSet, MyOcr, MonthDay

class DigitLines:
    def __init__(self, stem: str, direc: Path=input_dir_root, ext='.png', year=0, month=0):
        self.path_set = PathSet(parent=direc, stem=stem, ext=ext)
        self.my_ocr = MyOcr(year=year, month=month)
        self.digit_lines: Sequence[Any]|None = None
    def run_ocr(self) -> Sequence[Any]|None:
        result = self.my_ocr.run_ocr(path_set=self.path_set, lang='jpn+eng', builder_class=DigitLineBoxBuilder, layout=7) 
        match result:
            case Success(self.digit_lines):
                return self.digit_lines
            case Failure(_):
                return None #raise ValueError(f"Failed to run OCR and get data!")


class TxtLines:
    """Enhance txt_lines(Sequence[LineBox]) with img_pathset and MyOcr"""

    app_type = AppType.NUL

    def __init__(self, txt_lines: Sequence[LineBox], img_pathset: PathSet, my_ocr: MyOcr = MyOcr()):
        self.my_ocr = my_ocr
        if not img_pathset.exists():
            raise FileNotFoundError(f"Image path {img_pathset} does not exist!")
        self.txt_lines = txt_lines
        self.img_pathset = img_pathset
    def title(self, n=0):
        return self.txt_lines[n].content.replace(' ', '')
    def wages(self):
        raise NotImplementedError("wages not impremented!")


class TTxtLines(TxtLines):
    """
    Class to handle text lines from T app..
    """

    app_type = AppType.T

    def __init__(self, txt_lines: Sequence[LineBox], img_pathset: PathSet, my_ocr: MyOcr = MyOcr()):
        super().__init__(txt_lines, img_pathset=img_pathset, my_ocr=my_ocr)

    def wages(self):
        return self.my_ocr.t_wages(self.txt_lines)

    def get_date(self) -> MonthDay:
        for n, txt_line in enumerate(self.txt_lines):
            if txt_line.content.replace(' ', '').startswith('業務開始'):
                break
        if n >= len(self.txt_lines) - 1:
            logger.error("No date found in txt_lines for stem: {}", self.img_pathset.stem)
            raise ValueError(f"No date found in txt_lines for stem: {self.img_pathset.stem}")
        date_position = self.txt_lines[n + 1].position
        date_position = date_position[0] + date_position[1]

        img_path = self.img_pathset #.parent / (self.img_pathset.stem + self.img_pathset.ext)
        abs_img_path = img_path.resolve()
        date_image = Image.open(abs_img_path).crop(date_position)

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
                else:
                    import unicodedata
                    normalized_value = unicodedata.normalize('NFKC', no_spc_value)
                    mt = re.match(r"(\d+)月(\d+)日", normalized_value)
                    if mt and len(mt.groups()) == 2:
                        month, day = mt.groups()
                        date = MonthDay(int(month), int(day))
                        return date
                    raise ValueError("No match string of date!")
            case Failure(_):
                logger.error("No date found in txt_lines for stem: {}", self.img_pathset.stem)
                raise ValueError(f"No date found in txt_lines for stem: {self.img_pathset.stem}")

    def get_title(self) -> str:
        for n, txt_line in enumerate(self.txt_lines):
            if txt_line.content.replace(' ', '') == 'この店舗の募集状況':
                break
        if n == 0:
            logger.error("No title found in txt_lines for stem: {}", self.img_pathset.stem)
            raise ValueError(f"No title found in txt_lines for stem: {self.img_pathset.stem}")
        xmax = max([p[1][0] for p in [s.position for s in self.txt_lines[:n]]])

        rec_pos = self.txt_lines[n].position
        title_box = (rec_pos[0][0] - (rec_pos[1][1] - rec_pos[0][1]) - 10, 15,
                     xmax + 5, rec_pos[0][1] - 5 )

        img_path = self.img_pathset.parent / (self.img_pathset.stem + self.img_pathset.ext)
        title_img = Image.open(str(img_path)).crop(title_box)

        title_img_dir = self.img_pathset.parent.parent / 'TMP'
        title_img_dir.mkdir(parents=True, exist_ok=True)
        title_img_fullpath = title_img_dir / f'title-{self.img_pathset.stem}.png'
        title_img.save(title_img_fullpath, format='PNG')
        logger.info("Saved title image: {}", title_img_fullpath)
        result = self.my_ocr.run_ocr(path_set=title_img_fullpath, lang='jpn', builder_class=pyocr.builders.LineBoxBuilder)
        match result:
            case Success(title_txt_lines):
                acc_title = ''.join([line.content.replace(' ', '') for line in title_txt_lines])
                return acc_title
            case Failure(_):
                logger.error("No text lines found in : {}", title_img_fullpath)
                raise ValueError(f"No text lines found in : {title_img_fullpath}")


class MTxtLines(TxtLines):

    def title(self, n=8):
        return ':'.join([self.txt_lines[i].content.replace(' ', '') for i in range(n - 3, n - 1)])

    def wages(self):
        return self.my_ocr.m_wages(self.txt_lines)

    def get_date(self) -> tuple[int, MonthDay]:
        img_pathset = self.img_pathset
        my_ocr = self.my_ocr
        n, date, hrs = my_ocr.check_date(app_type=AppType.M, txt_lines=self.txt_lines)
        if date:
            return n, date
        else: # Retry cropping the image..
            box_pos = [*self.txt_lines[n].position]
            box_pos = list(box_pos[0] + box_pos[1])
            box_pos[0] += box_pos[3] - box_pos[1] # remove leading emoji that is about the same width of the box height
            image = Image.open(img_pathset.parent / (img_pathset.stem + img_pathset.ext))
            if not image:
                logger.error("Failed to open image: {}", img_pathset)
                raise ValueError(f"Failed to open image: {img_pathset}")
            box_img = image.crop(tuple(box_pos))
            if not box_img:
                logger.error("Failed to crop box image: {}", box_pos)
                raise ValueError(f"Failed to crop box image: {box_pos}")
            tmp_img_dir = img_pathset.parent.parent / 'TMP'
            tmp_img_dir.mkdir(parents=True, exist_ok=True)
            box_img_fullpath = tmp_img_dir / f'{img_pathset.stem}.box.png'
            box_img.save(box_img_fullpath, format='PNG')
            logger.info("Saved box image: {}", box_img_fullpath)    
            box_result = my_ocr.run_ocr(box_img_fullpath, lang='jpn', builder_class=pyocr.builders.TextBuilder, layout=7, opt_img=box_img)
            match box_result:
                case Success(value):
                    _n, date, hrs = my_ocr.check_date(app_type=AppType.M, txt_lines=[value]) # len(txt_lines) == 1
                    if date is None:
                        logger.error("Failed to get date from box image: {}", box_pos)
                        raise ValueError(f"Failed to get date from box image: {box_pos}")
                    logger.info("Date by run_ocr with TextBuilder and cropped image: {}", date)
                    return n, date
                case Failure(_):
                    logger.error("Failed to run OCR on box image: {}", box_pos)
                    raise ValueError(f"Failed to run OCR on box image: {box_pos}")
