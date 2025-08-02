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
#from ipdb import set_trace as breakpoint
from dotenv import load_dotenv
load_dotenv(override=True)
from PIL import Image, ImageDraw, ImageEnhance
import pyocr
import pyocr.builders
from pyocr.builders import LineBox
from returns.pipeline import is_successful, UnwrapFailedError
import logbook

logbook.StreamHandler(sys.stdout,
	format_string='{record.time:%Y-%m-%d %H:%M:%S.%f} {record.level_name} {record.filename}:{record.lineno}: {record.message}').push_application()

logger = logbook.Logger(__file__)
logger.level = logbook.INFO
from app_type import AppType
@dataclass
class MonthDay:
    month: int
    day: int
    @property
    def as_float(self):
        return float(f"{self.month}.{self.day:02}")

def get_date_split(line_box: pyocr.builders.LineBox)-> Union[MonthDay, None]:
    content = line_box.content.split()
    if len(content) > 3:
        if (content[1] == '月') and (content[3] == '日'):
        #raise ValueError("Not 月日!")
            return MonthDay(month=int(content[0]), day=int(content[2]))

DATE_PATT = re.compile(r"(\d+)月(\d+)日")
@safe
def get_date_T_LineBox(line_box: pyocr.builders.LineBox)-> Union[MonthDay, None]:
    content = line_box.content.replace(' ', '')
    result = DATE_PATT.search(content)
    if result:
        month, day = result.groups()
        return MonthDay(month=int(month), day=int(day)) 

'''def next_gyoumu(txt_lines: Sequence[pyocr.builders.LineBox]):
    lines_len = len(txt_lines)
    for n, tx in enumerate(txt_lines):
        joined_tx = ''.join([t.strip() for t in tx.content])
        if joined_tx[0:4] == '業務開始':
            return n + 1
    #return txt_lines[n + 1] #.content '''


from collections import namedtuple
from dataclasses import dataclass
@dataclass
class PathSet:
    parent: Path
    stem: str
    ext: str
    def stem_without_delim(self, delim: str=''):
        return ''.join([s for s in self.stem if s!= delim])
    def exists(self):
        return (self.parent / (self.stem_without_delim('') + self.ext)).exists()
    def open(self):
        return (self.parent / (self.stem_without_delim('') + self.ext)).open()
    def to_path(self):
        return self.parent / (self.stem_without_delim('') + self.ext)


APP_TYPE_TO_STEM_END = MappingProxyType({
    AppType.T: ".co.taimee",
    AppType.M: ".mercari.work.android"
})


class OCRError(Exception):
    pass

class MDateError(OCRError):
    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self):
        return f"MDateError: {self.args[0]}"

from getpass import getpass
from subprocess import Popen, PIPE
import shlex
def load_tools():
    password = getpass("Going to install Tesseract tools. Enter your sudo user password:(Just 'Enter' if no password)")
    # sudo requires the flag '-S' in order to take input from stdin
    with_pw = f"-S" if password else ''
    cmds_txt = f'''
        sudo {with_pw} apt install tesseract-ocr -y
        sudo {with_pw} apt install libtesseract-dev -y
        sudo {with_pw} apt install tesseract-ocr-jpn -y
        '''
    cmd_lines = [txt.strip() for txt in cmds_txt.split('\n') if txt.strip()]
    for cmd_line in cmd_lines:
        cmds = shlex.split(cmd_line)
        proc = Popen(cmds, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        if password:
            proc.communicate(password.encode())
        print("Installing tools: {}".format(proc))
        while proc.poll() is None:
            print('status:', proc.stdout.readline().decode().strip())

class MyOcr:
    from path_feeder import input_dir_root # input_ext
    tools = pyocr.get_available_tools()

    if len(tools) == 0:
        load_tools()
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            raise ValueError("No OCR tool found")
    tool = tools[0]
    assert tool is not None, "No OCR tool found!"

    #input_dir = input_dir # Path(os.environ['SCREEN_BASE_DIR'])
    delim = ' '

    @classmethod
    def get_tool_name(cls):
        return(cls.tool.get_name())    # 'Tesseract (sh)'

    @classmethod
    def get_app_type(cls, txt_lines: Sequence[LineBox]):
        for txt_line in txt_lines:
            if txt_line.content.replace(' ', '').find('おしごと詳細') >= 0:
                return AppType.M
        return AppType.NUL # TODO: check for 'TM'
    @classmethod
    def get_title(cls, app_type: AppType, txt_lines: Sequence[LineBox], n: int):
        match app_type:
            case AppType.T:
                return cls.t_title(txt_lines=txt_lines)
            case AppType.M:
                return cls.m_title(txt_lines, n)
    @classmethod
    def get_wages(cls, app_type: AppType, txt_lines: Sequence[LineBox]):
        match app_type:
            case AppType.T:
                return cls.t_wages(txt_lines=txt_lines)
            case AppType.M:
                for n in range(len(txt_lines)):
                    if txt_lines[n].content.replace(' ', '').find('このおしごとの給与'
        ) >= 0:
                        wages = int(''.join([i for i in txt_lines[n + 1].content.replace(' ', '') if i in '0123456789']))
                        if wages in range(1000, 9999):
                            return wages
                        else:
                            raise ValueError("Improper wage!")
            case _: # TODO: check for 'TM'
                raise ValueError("Undefined AppType!")

    M_DATE_PATT = [re.compile(r"(\d\d:\d\d)"),
        re.compile(r"(\d+)/(\d+)"),
        re.compile(r"時\s*間$")]
    @classmethod
    def check_date(cls, app_type: AppType, txt_lines: Sequence[LineBox|str])->tuple[int, MonthDay|None,Sequence[str]]:
        match app_type:
            case AppType.M:
                for n in range(len(txt_lines)):
                    cntnt = txt_lines[n].content if isinstance(txt_lines[n], LineBox) else txt_lines[n] #.content.replace(' ', '') 
                    if MyOcr.M_DATE_PATT[-1].search(cntnt):
                      if (hours:=MyOcr.M_DATE_PATT[0].findall(cntnt)):
                        m_d = MyOcr.M_DATE_PATT[1].match(cntnt)
                        if m_d:
                            grps = m_d.groups()
                            date = MonthDay(int(grps[0]), int(grps[1]))
                            return n, date, hours
                        else:
                            return n, None, hours
                raise MDateError(f"Could not resolve date! AppType.M txt_lines!:{txt_lines}")
            case AppType.T:
                raise NotImplementedError("Not implemented yet!")
            case _:
                raise ValueError("Undefined AppType!")
    @classmethod
    @safe
    def get_date(cls, app_type: AppType, txt_lines: Sequence[LineBox]|str):
        match app_type:
            case AppType.M:
                for n in range(len(txt_lines)):
                    cntnt = txt_lines[n].content.replace(' ', '') # concatenated
                    if cntnt.endswith('時間') and (hours:=MyOcr.M_DATE_PATT[0].findall(cntnt)):
                        m_d = MyOcr.M_DATE_PATT[1].match(cntnt)
                        if m_d:
                            grps = m_d.groups()
                            date = MonthDay(int(grps[1]), int(grps[2]))
                            return n, date, hours
                        else:
                            raise MDateError(f"Could not find month and day: AppType.M txt_lines!: {txt_lines}")
                raise MDateError(f"Could not resolve date! AppType.M txt_lines!: {txt_lines}")
            case AppType.T:
                for n, txt_line in enumerate(txt_lines):
                    content = txt_line.content.replace(' ', '') if isinstance(txt_line, LineBox) else txt_line
                    if '業務開始' in content:
                        break
                result = get_date_T_LineBox(txt_lines[n + 1]) # next line
                match(result):
                    case Success(date):
                        return n, date
                    case Failure(_):
                        raise ValueError("No date found in T!")
            case _:
                raise ValueError("Undefined AppType!")

    def __init__(self, month=0, year=0):
        from path_feeder import get_year_month
        self.date = get_year_month(year=year, month=month)
        #self.month = month
        #self.input_dir = MyOcr.input_dir_root
        #from path_feeder import PathFeeder
        #self.path_feeder = PathFeeder(input_dir=MyOcr.input_dir_root, type_dir=False, month=month)
        self.txt_lines: Sequence[pyocr.builders.LineBox] | None = None
        self.image: Image.Image | None = None
    @property
    def input_dir(self):
        return MyOcr.input_dir_root / str(self.date.year) / f'{self.date.month:02}' / 'png'
    '''def each_path_set(self):
        for stem in self.path_feeder.feed(delim=self.delim, padding=False):
            yield PathSet(self.path_feeder.dir, stem, self.path_feeder.ext)'''
    @safe
    def run_ocr(self, path_set: Path | PathSet, lang='eng+jpn', delim='',
                builder_class=pyocr.builders.LineBoxBuilder, layout=3, opt_img: Image.Image|None=None)-> Sequence[pyocr.builders.LineBox]|str:#, Exception]: # Digit
        #stem_without_delim = ''.join([s for s in path_set[1] if s!= self.delim])
        fullpath = path_set if isinstance(path_set, Path) else path_set.parent / (path_set.stem_without_delim(delim) + path_set.ext)
        if not opt_img:
            img = Image.open(fullpath).convert('L')
            enhancer= ImageEnhance.Contrast(img)
            self.image = enhancer.enhance(2.0)
        logger.info("Start to OCR: file: {}, file-size: {}, image-width: {}, image-height: {}", fullpath, fullpath.stat().st_size, opt_img.width if opt_img else self.image.width, opt_img.height if opt_img else self.image.height)
        txt_lines = self.tool.image_to_string(
            opt_img or self.image,
            lang=lang,
            builder=builder_class(tesseract_layout=layout)
        )
        if isinstance(txt_lines, list) and txt_lines:
            self.txt_lines = txt_lines
            logger.info("OCR result: {} lines", len(txt_lines))
        elif isinstance(txt_lines, str) and txt_lines:
            logger.info("OCR result: {}", txt_lines)
        else:
            logger.warning("OCR result is None!")
        return txt_lines
        # raise ValueError("PYOCR run failed!")
        
    
    def draw_boxes(self):
        if not self.txt_lines:
            raise ValueError('txt_lines is None!')
        if not self.image:
            raise ValueError('Image is None!')
        draw = ImageDraw.Draw(self.image)
        for line in self.txt_lines:
            cood = line.position[0] + line.position[1]
            draw.rectangle(cood, outline=0x55)
        self.image.show()
        return self
    
    def endswith(self, pattern: str):
        for file in self.input_dir_root.iterdir():
            if file.is_file and file.stem.endswith(pattern):
                yield file


    @classmethod
    def t_title(cls, txt_lines: Sequence[LineBox]):
        return txt_lines[1].content.replace(' ', '')

    @classmethod
    def m_title(cls, txt_lines: Sequence[LineBox], n: int):
        return ''.join([txt_lines[i].content.replace(' ', '') for i in range(n - 3, n - 1)])


    @classmethod
    def m_wages(cls, txt_lines: Sequence[LineBox], n_cc=['%d' % c for c in range(10)]):
        found = False
        for ln in txt_lines:
            if '合計' in ln.content.replace(' ', ''):
                found = True
                break
        if not found:
            raise ValueError(f"'合計' is not found!:{txt_lines}")
        content = ln.content
        nn = []
        for n in content:
            if n in n_cc:
                nn.append(n)
        return int(''.join(nn))
    @classmethod
    def t_wages(cls, txt_lines: Sequence[LineBox]):
        content = txt_lines[-1].content
        content_num = ''.join(re.findall(r'\d+', content)) # ''.join([n for n in content if '0123456789'.index(n) >= 0])
        try:
            num = int(content_num)
            if not (1000 < num < 9999):
                raise ValueError(f"Unexpected value: {num}")
            return num
        except ValueError as err:
            raise ValueError(f"Failed to convert into an integer: {content_num}\n{err}")
    @classmethod
    def w_wages(cls, txt_lines: Sequence[LineBox]):
        raise NotImplementedError("Not implemented yet!")

