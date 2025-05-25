from typing import Sequence
from contextlib import closing
from enum import IntEnum
from types import MappingProxyType
import re
from typing import Sequence, Callable, Any
from pprint import pp
from pathlib import Path
from dataclasses import dataclass
import pickle

import pandas
from dotenv import load_dotenv
load_dotenv()
from PIL import Image, ImageDraw, ImageEnhance
import pyocr
import pyocr.builders
from pyocr.builders import LineBox
from returns.pipeline import is_successful, UnwrapFailedError
import loguru # logging
logger = loguru.logger # logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.remove()
import sys
#logger.add(sys.stderr, level="DEBUG")
logger.add(sys.stdout, level="INFO")
logger.add("WARNING.log", level="WARNING")
from app_type import AppType
@dataclass
class Date:
    month: int
    day: int
    @property
    def as_float(self):
        return float(f"{self.month}.{self.day:02}")

from typing import Union
def get_date(line_box: pyocr.builders.LineBox)-> Union[Date, None]:
    content = line_box.content.split()
    if len(content) > 3:
        if (content[1] == '月') and (content[3] == '日'):
        #raise ValueError("Not 月日!")
            return Date(month=int(content[0]), day=int(content[2]))

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
    path: Path
    stem: str
    ext: str
    def stem_without_delim(self, delim: str):
        return ''.join([s for s in self.stem if s!= delim])

APP_TYPE_TO_STEM_END = MappingProxyType({
    AppType.T: ".co.taimee",
    AppType.M: ".mercari.work.android"
})


class OCRError(Exception):
    pass

from returns.result import safe, Result, Failure, Success
class MDateError(OCRError):
    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self):
        return f"MDateError: {self.args[0]}"
class MyOcr:
    from path_feeder import input_dir_root, input_ext
    tools = pyocr.get_available_tools()
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
        re.compile(r"(\d+?)/(\d+?)"),
        re.compile(r"時\s*間$")]
    @classmethod
    def check_date(cls, app_type: AppType, txt_lines: Sequence[LineBox|str])->tuple[int, Date|None,Sequence[str]]:
        match app_type:
            case AppType.M:
                for n in range(len(txt_lines)):
                    cntnt = txt_lines[n].content if isinstance(txt_lines[n], LineBox) else txt_lines[n] #.content.replace(' ', '') 
                    if MyOcr.M_DATE_PATT[-1].search(cntnt):
                      if (hours:=MyOcr.M_DATE_PATT[0].findall(cntnt)):
                        m_d = MyOcr.M_DATE_PATT[1].match(cntnt)
                        if m_d:
                            grps = m_d.groups()
                            date = Date(int(grps[0]), int(grps[1]))
                            return n, date, hours
                        else:
                            return n, None, hours
                raise MDateError(f"Could not resolve date! AppType.M txt_lines!:{txt_lines}")
            case AppType.T:
                raise NotImplementedError("Not implemented yet!")
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
                            date = Date(int(grps[1]), int(grps[2]))
                            return n, date, hours
                        else:
                            raise MDateError(f"Could not find month and day: AppType.M txt_lines!: {txt_lines}")
                raise MDateError(f"Could not resolve date! AppType.M txt_lines!: {txt_lines}")
            case AppType.T:
                # n_gyoumu = next_gyoumu(txt_lines)
                date = None
                for n, txt_line in enumerate(txt_lines):
                    date = get_date(txt_line)
                    if date:
                        break
                if not date:
                    raise ValueError("No date found in T!")
                return n, date
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
        fullpath = path_set if isinstance(path_set, Path) else path_set.path / (path_set.stem_without_delim(delim) + path_set.ext)
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
            logger.info("OCR result: %d lines", len(txt_lines))
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
        '''stop = 'この 店 舗 の 募集 状況'.replace(' ', '')
        lines = []
        for line in txt_lines:
            if (content:=''.join(line.content.split())) == stop:
                break
            lines.append(content)
        return ';'.join(lines)'''
    @classmethod
    def m_title(cls, txt_lines: Sequence[LineBox], n: int):
        return ''.join([txt_lines[i].content.replace(' ', '') for i in range(n - 3, n - 1)])

    '''def get_date(self, app_tpe: AppType):
        if not self.txt_lines:
            raise bb ValueError('`txt_lines` is None!')
        match
        n_gyoumu = next_gyoumu(self.txt_lines)
        return get_date(n_gyoumu)'''
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


class TTxtLines:
    TITLE = 1
    def __init__(self, txt_lines: Sequence[LineBox]):
        self.txt_lines = txt_lines

    def title(self, n=0):
        return self.txt_lines[1].content.replace(' ', '')
    def wages(self):
        return MyOcr.t_wages(self.txt_lines)

class MTxtLines(TTxtLines):
    def title(self, n: int):
        return ''.join([self.txt_lines[i].content.replace(' ', '') for i in range(n - 3, n - 1)])
    def wages(self):
        return MyOcr.m_wages(self.txt_lines)

class Main:
    import txt_lines_db
    def __init__(self, app=AppType.NUL, db_fullpath=txt_lines_db.sqlite_fullpath(), my_ocr=MyOcr()):  
        self.my_ocr = my_ocr
        #self.img_dir = self.my_ocr.input_dir
        #month = self.my_ocr.month
        #img_parent_dir = self.img_dir.parent
        self.app = app # tm
        # txt_lines_db = TxtLinesDB(img_parent_dir=img_parent_dir)
        self.conn = Main.txt_lines_db.connect(db_fullpath=db_fullpath)
        self.tbl_name = Main.txt_lines_db.get_table_name(self.my_ocr.date.month)
        Main.txt_lines_db.create_tbl_if_not_exists(self.tbl_name)
    
    @property
    def img_dir(self):
        return self.my_ocr.input_dir

    def db_path_feeder(self, app: AppType, feeder_dict={}):
        if feeder:=feeder_dict[app]:
            return feeder
        from path_feeder import DbPathFeeder
        feeder_dict[app] = feeder = DbPathFeeder(month=self.month, app_type=app)
        return feeder

    @property
    def month(self):
        return self.my_ocr.date.month

    def sum_wages(self):
        with closing(self.conn.cursor()) as cur:
            sum_q = f"SELECT SUM(wages) from `{self.tbl_name}`"
            result = cur.execute(sum_q)
            if result:
                return [r[0] for r in result][0]
    def get_OCRed_files(self):
        qry = f"SELECT stem from `{self.tbl_name}` order by stem;"
        with closing(self.conn.cursor()) as cur:
            cur.execute(qry)
            all = cur.fetchall()
            if all:
                return [a[0] for a in all]

    def get_existing_days(self, app_type=AppType.NUL)-> dict[AppType, list[int]] | list[int] | None:
        if not app_type.value:
            from collections import defaultdict
            qry = f"SELECT app, day from `{self.tbl_name}` order by app, day;"
            with closing(self.conn.cursor()) as cur:
                cur.execute(qry)
                all = cur.fetchall()
                if all:
                    r_dict = defaultdict(list)
                    for app, day in all:
                        app_type = AppType(app)
                        r_dict[app_type].append(day)
                    return r_dict
                else:
                    return
        qry = f"SELECT day from `{self.tbl_name}` where app = ?;"
        prm = (app_type.value, ) # date.day)
        with closing(self.conn.cursor()) as cur:
            result = cur.execute(qry, prm)
            if result:
                return [r[0] for r in result]

    def add_image_files_as_txt_lines(self, app_type: AppType, ext='.png'):
        if app_type == AppType.NUL:
            raise ValueError('Needs not AppType.NUL param.!')
        ocr_done = []
        glob_patt = '*' + APP_TYPE_TO_STEM_END[app_type] + ext
        logger.debug(f"glob_patt: {glob_patt}")
        for img_file in self.img_dir.glob(glob_patt):
            stem = img_file.stem
            parent = self.my_ocr.input_dir
            path_set = PathSet(parent, stem, ext)
            result = self.my_ocr.run_ocr(path_set=path_set, delim='')
            match result:
                case Success(value):
                    txt_lines = value # result.unwrap()
                case Failure(_): # if not is_successful(result): # type: ignore
                    raise ValueError(f"Failed to run OCR!")#Unable to extract from {path_set}")
            result = self.my_ocr.get_date(app_type=app_type, txt_lines=txt_lines)
            match result:
                case Success(value):
                    n, date = value # result.unwrap()
                case Failure(_): # if not is_successful(result): # type: ignore
                    raise ValueError(f"Failed to run OCR!")#Unable to extract from {path_set}")
            
            existing_day_list = self.get_existing_days(app_type=app_type)
            if existing_day_list and (date.day in existing_day_list):
                print(f"Day {date.day} of App {app_type} exists.")
            else:
                wages = self.my_ocr.get_wages(app_type=app_type, txt_lines=txt_lines)
                title = self.my_ocr.get_title(app_type, txt_lines, n)
                pkl = pickle.dumps(txt_lines)
                with closing(self.conn.cursor()) as cur:
                    cur.execute(f"INSERT INTO `{self.tbl_name}` VALUES (?, ?, ?, ?, ?, ?);", (app_type.value, date.day, wages, title, stem, pkl))
                for line in txt_lines:
                    logger.info("txt_lines: {}", line.content)
                self.conn.commit()
                ocr_done.append((app_type, date))
        return ocr_done
    from returns.pipeline import is_successful
    def add_image_files_into_db(self, ext='.png'):
        existing_day_dict = self.get_existing_days()
        ocr_done = []
        ocred_files = self.get_OCRed_files()
        stemend_to_apptype = {APP_TYPE_TO_STEM_END[app_type]:app_type for app_type in AppType if app_type > 0}
        def is_screenshot_file(stem: str):
            for stemend, apptype in stemend_to_apptype.items():
                if stem.endswith(stemend):
                    return apptype
            #return AppType.NUL
        for img_file in self.img_dir.glob('*' + ext):
            if ocred_files and (img_file.stem in ocred_files):
                continue
            app_type = is_screenshot_file(img_file.stem)
            if not app_type:#== AppType.NUL:
                continue
            stem = img_file.stem
            parent = self.my_ocr.input_dir
            path_set = PathSet(parent, stem, ext)
            result = self.my_ocr.run_ocr(path_set=path_set, delim='')
            match result:
                case Success(value):
                    txt_lines = value # result.unwrap()
                case Failure(_):
                    raise ValueError(f"Failed to run OCR of {path_set}")
            n, date = self.my_ocr.get_date(app_type=app_type, txt_lines=txt_lines)
            existing_day_list = existing_day_dict[app_type] if (existing_day_dict and app_type in existing_day_dict) else None
            if existing_day_list:
                if date.day in existing_day_list:
                    print(f"Day {date.day} of App {app_type} exists.")
            else:
                wages = self.my_ocr.get_wages(app_type=app_type, txt_lines=txt_lines)
                title = self.my_ocr.get_title(app_type, txt_lines, n)
                pkl_file_fullpath = self.img_dir / (stem + '.pkl')
                with pkl_file_fullpath.open('wb') as wf:
                    pkl = pickle.dump(txt_lines, wf)
                with closing(self.conn.cursor()) as cur:
                    cur.execute(f"INSERT INTO `{self.tbl_name}` VALUES (?, ?, ?, ?, ?, ?);", (app_type.value, date.day, wages, title, stem, None))
                for line in txt_lines:
                    pp(line.content)
                self.conn.commit()
                ocr_done.append((app_type, date))
        return ocr_done
    def add_image_file_without_content_into_db(self, app_type: AppType, stem: str, date: Date, wages=None, title=None, pkl=None):
        with closing(self.conn.cursor()) as cur:
            cur.execute(f"INSERT INTO `{self.tbl_name}` VALUES (?, ?, ?, ?, ?, ?);", (app_type.value, date.day, wages, title, stem, pkl))
        self.conn.commit()

    def ocr_result_into_db(self, app_type_list: list[AppType]|None=None, limit=62, test=False):
        if not app_type_list:
            app_type_list = [e for e in list(AppType) if e != AppType.NUL]  
        assert limit > 0 
        count = 0
        with closing(self.conn.cursor()) as cur:
            for app_type in app_type_list:
                stem_end_patt = APP_TYPE_TO_STEM_END[app_type]
                glob_patt = "*" + stem_end_patt + '.png'
                for file in self.my_ocr.input_dir.glob(glob_patt):
                    if file.stem in self.get_OCRed_files():
                        continue
                    app = app_type.value
                    path_set = PathSet(self.my_ocr.input_dir, file.stem, ext=self.my_ocr.input_ext)
                    result = self.my_ocr.run_ocr(path_set)
                    match result:
                        case Success(value):
                            txt_lines = value # result.unwrap()
                        case Failure(_):
                            raise ValueError("Failed to run OCR!", path_set)
                    if app_type == AppType.M:
                        n, date, hrs = self.my_ocr.check_date(app_type=app_type, txt_lines=txt_lines)
                        if date is None:
                            box_pos = [*txt_lines[n].position]
                            box_pos = box_pos[0] + box_pos[1]
                            box_pos = [box_pos[0] + box_pos[3] - box_pos[1], box_pos[1], box_pos[2], box_pos[3]]
                            box_img = self.my_ocr.image.crop(box_pos) if self.my_ocr.image else None
                            if not box_img:
                                logger.error("Failed to crop box image: {}", box_pos)
                                raise ValueError(f"Failed to crop box image: {box_pos}")
                            box_result = self.my_ocr.run_ocr(path_set, lang='eng+jpn', builder_class=pyocr.builders.TextBuilder, layout=7, opt_img=box_img)
                            match box_result:
                                case Success(value):
                                    n, date, hrs = self.my_ocr.check_date(app_type=app_type, txt_lines=[value])
                                    if date is None:
                                        logger.error("Failed to get date from box image: {}", box_pos)
                                        raise ValueError(f"Failed to get date from box image: {box_pos}")
                                    logger.info("Date by run_ocr with TextBuilder and cropped image: {}", date)
                                case Failure(_):
                                    logger.error("Failed to run OCR on box image: {}", box_pos)
                                    raise ValueError(f"Failed to run OCR on box image: {box_pos}")
                            if test:
                                debug_dir = self.my_ocr.input_dir / 'DEBUG'
                                debug_dir.mkdir(parents=True, exist_ok=True)
                                debug_fullpath = debug_dir / (file.stem + '.dbg.png') 
                                box_img.save(debug_fullpath)
                                logger.info("Saved debug image: {}", debug_fullpath)
                    elif app_type == AppType.T:
                        result = self.my_ocr.get_date(app_type=app_type, txt_lines=txt_lines)
                        match result:
                            case Success(value):
                                n, date = value # result.unwrap()
                            case Failure(_): # if not is_successful(result): # type: ignore
                                logger.info("Failed to get date from {} in lang=jpn+eng, try to run_ocr in lang=jpn..", file)
                                e_result = self.my_ocr.run_ocr(path_set, lang='eng',)
                                match e_result:
                                    case Success(value):
                                        d_result = self.my_ocr.get_date(app_type=app_type, txt_lines=value)
                                        match d_result:
                                            case Success(value):
                                                n, date = value
                                            case Failure(_):
                                                logger.error("Failed to get date from txt_lines:{} in lang=eng!", txt_lines)
                                                raise ValueError(f"Failed to get date from file: {file} in lang=eng!")
                                    # if not is_successful(e_result): # type: ignore
                                    case Failure(_):
                                        raise ValueError("Failed to run OCR in lang=eng!", path_set)
                    else:
                        raise NotImplementedError(f"Undefined AppType: {app_type}!")
                    exists_sql = f"SELECT day, app, stem FROM `{self.tbl_name}` WHERE day = {date.day} AND app = {app};"
                    cur.execute(exists_sql)
                    if (one:= cur.fetchone()):
                        logger.info("Day {} of App {} already exists in DB: {}", date.day, app, one)
                        old_app = int(one[1])
                        old_stem = one[2]
                        if old_app == app and old_stem != file.stem:
                            old_pkl_path = self.get_pkl_path(old_stem)
                            if old_pkl_path.exists():
                                logger.warning("Old pkl file exists: {}", old_pkl_path)
                                with old_pkl_path.open('rb') as rf:
                                    old_txt_lines = pickle.load(rf)
                                    box_pos = [*old_txt_lines[n + 1].position]
                                    box_pos = box_pos[0] + box_pos[1]
                                old_file = self.get_replaced_stem_file(file, old_stem)
                                if not old_file.exists():
                                    raise ValueError(f"Old file does not exist: {old_file}")
                                image = Image.open(old_file)
                                if not image:
                                    logger.error("Failed to open old file: {}", old_file)
                                    raise ValueError(f"Failed to open old file: {old_file}")
                                box_img = image.crop(box_pos)
                                box_result = self.my_ocr.run_ocr(path_set, lang='eng+jpn', builder_class=pyocr.builders.TextBuilder, layout=7, opt_img=box_img)
                                match box_result:
                                    case Success(value):
                                        breakpoint()
                                        logger.debug("date-part OCR result: {}", value)
                                        no_spc_value = value.replace(' ', '')
                                        mt = re.match(r"(\d+)月(\d+)日", no_spc_value)
                                        if mt:
                                            month, day = mt.groups()
                                            box_date = Date(int(month), int(day))
                                            assert box_date.month == date.month, f"Month from box image: {box_date.month} does not match: {date.month}"
                                            if box_date.day != date.day:
                                                logger.info("Date from box image: {} is different from DB ", box_date.day)
                                                backup_sql = f"SELECT wages, title, stem FROM `{self.tbl_name}` WHERE day = {date.day} AND app = {app};"
                                                cur.execute(backup_sql)
                                                old_wages, old_stem, old_title = cur.fetchone()
                                                replace_sql = f"UPDATE `{self.tbl_name}` SET stem = ?, wages = ?, title = ? WHERE day = {date.day} AND app = {app};"
                                                cur.execute(replace_sql, (file.stem, wages, title))
                                                logger.info("REPLACE: {}", (file.stem, wages, title))
                                            else:
                                                logger.error("Date from box image does not match: {}, expected: {}", box_date, date)
                                                raise ValueError(f"Date from box image does not match: {box_date}, expected: {date}")
                                            replace_sql = f"UPDATE `{self.tbl_name}` SET stem = ?, wages = ?, title = ? WHERE day = {date.day} AND app = {app};"
                                        else:
                                            logger.error("Failed to get date from value: {}", no_spc_value)
                                            raise ValueError(f"Failed to get date from value: {no_spc_value}")
                                        breakpoint()
                                    case Failure(_):
                                        logger.error("Failed to run OCR on box image: {}", box_pos)
                                        raise ValueError(f"Failed to run OCR on box image: {box_pos}")


                                if test:
                                    debug_dir = self.my_ocr.input_dir.parent / 'DEBUG'
                                    debug_dir.mkdir(exist_ok=True)
                                    debug_fullpath = debug_dir / (file.stem + f'({box_pos}).png') 
                                    data_image.save(debug_fullpath)
                                    logger.info("Saved debug image: {}", debug_fullpath)
                                    breakpoint()
                            else:
                                logger.warning("Old pkl file does not exist: {}", old_pkl_path)
                        assert old_stem != file.stem
                        raise ValueError(f"Day {date.day} of App {app} already exists in DB with stem: {old_stem}: new stem:{file.stem}!")
                    tm_txt_lines = {AppType.T: TTxtLines, AppType.M: MTxtLines}[app_type](txt_lines)
                    wages = tm_txt_lines.wages()
                    title = tm_txt_lines.title(n)

                    pkl_fullpath = self.get_pkl_path(file.stem) #pkl_dir / (file.stem + '.pkl')
                    with pkl_fullpath.open('wb') as wf:
                        pickle.dump(txt_lines, wf)
                    insert_sql = f"INSERT INTO `{self.tbl_name}` VALUES ({app}, {date.day}, ?, ?, ?, ?)" 
                    cur.execute(insert_sql, (wages, title, file.stem, None))
                    logger.info("INSERT: {}", (app, date.day, wages, title, file.stem))
                    count += 1
                    limit -= 1
                    if limit <= 0:
                        logger.info("Limit reached: %d", limit)
                        break
        self.conn.commit()
        return count

    def get_pkl_path(self, stem: str):
        """Get the path of the pkl file."""
        pkl_dir = self.img_dir.parent / 'pkl'
        pkl_dir.mkdir(parents=True, exist_ok=True)
        return pkl_dir / (stem + '.pkl')
    def get_replaced_stem_file(self, file: Path, new_stem: str):
        """Replace the stem of the file with the new stem."""
        return file.parent / (new_stem + file.suffix)

    def save_as_csv(self):
        #conn = sqlite3.connect(db_file, isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES)
        table = f"text_lines-{self.my_ocr.date.month:02}"
        sql = f"SELECT * FROM `{table}`"
        db_df = pandas.read_sql_query(sql, self.conn)
        output_path = self.img_dir# / str(self.my_ocr.date.year) / f"{self.my_ocr.date.month:02}"
        assert output_path.exists()
        output_fullpath = output_path / (table + '.csv')
        db_df.to_csv(str(output_fullpath), index=False)

from contextlib import closing
from pickle import load
def edit_title(month: int, day: int):
    from path_feeder import DbPathFeeder
    feeder = DbPathFeeder(month=month)
    with closing(feeder.conn.cursor()) as cur:
        sql = f"SELECT stem, day, title FROM 'text_lines-{month:02}' WHERE day = {day}"
        cur.execute(sql)
        rr = cur.fetchone()
        stem = rr[0]
        db_day = rr[1]
        db_title = rr[2]
    assert day == db_day
    pkl_fullpath = feeder.dir / (stem + '.pkl')
    txt_lines = load(pkl_fullpath.open('rb'))
    title = txt_lines[0].content.replace(' ', '')
    yn = input(f"Replace '{db_title}' to '{title}'?(y/n):")
    if yn.lower()[0] == 'y':
        sql = f"UPDATE 'text_lines-{month:02}' SET title = ? WHERE day = {day}"
        with closing(feeder.conn.cursor()) as cur:
            rr = list(cur.execute(sql, (title,)))
        feeder.conn.commit()
def edit_wages(month: int, app=AppType.T):
    assert app != AppType.NUL
    from path_feeder import DbPathFeeder
    feeder = DbPathFeeder(month=month)
    with closing(feeder.conn.cursor()) as cur:
        sql = f"SELECT stem, day FROM 'text_lines-{month:02}' WHERE wages IS NULL"
        stem_day_list = list(cur.execute(sql))
    if stem_day_list:
        for stem, day in stem_day_list:
            pkl_fullpath = feeder.dir / (stem + '.pkl')
            txt_lines = load(pkl_fullpath.open('rb'))
            wages = MyOcr.t_wages(txt_lines) if app == AppType.T else MyOcr.w_wages(txt_lines)
            yn = input(f"Replace '{day}' in '{stem}' to '{wages}'?(y/n):")
            if yn.lower()[0] == 'y':
                table = f'text_lines-{month:02}'
                update_sql = f"UPDATE `{table}` SET wages = {wages} WHERE day = {day} AND `app` = {app.value}"
                with closing(feeder.conn.cursor()) as cur:
                    cur.execute(update_sql)
                    check_sql = f"SELECT wages FROM `{table}` where day = {day} AND `app` = {app.value}"
                    cur.execute(check_sql)
                    row = cur.fetchone()
                    assert row[0] == wages
                    print(f"Wages of {day=} is Updated as {wages=} in table `{table=}`.")
                    feeder.conn.commit()
def run_ocr(month: str, limit=62, app_type: AppType = AppType.NUL, test=False):
    """Run OCR and save result into DB."""
    m = int(month)
    my_ocr = MyOcr(month=m)
    main = Main(my_ocr=my_ocr, app=app_type)
    main.ocr_result_into_db(limit=limit, test=test)

import click
@click.group()
def cli():
    pass
@cli.command()
@click.argument('month')
def ocr(month: str):
    """Run OCR and save result into DB."""
    run_ocr(month)

class FunctionItem:
    def __init__(self, title: str, func: Callable | None, kwargs: dict[str, Any]={}):
        self.title = title
        self.func = func
        self.kwargs = kwargs
    def exec(self):
        if self.func:
            self.func(**self.kwargs) #if self.kwargs else self.func()
def get_options(month=0):
    my_ocr = MyOcr(month=month) 
    main = Main(my_ocr=my_ocr)
    return [
        FunctionItem('None', None),
        FunctionItem('save OCR result into DB', main.ocr_result_into_db),

    ]
def run_main(options: Sequence[FunctionItem]):#=get_options(int(input("Month?:") or '0'))):
    for n, option in enumerate(options):
        print(f"{n}. {option.title}")
    choice = int(input(f"Choose(0 to {len(options)}):"))
    if choice:
        options[choice].exec()
if __name__ == '__main__':
    import sys
    cli()
    sys.exit(0)
