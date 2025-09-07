from typing import Sequence, override
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
from loguru import logger
logger.remove()
logger.add(sys.stderr, level='INFO', format="{time} | {level} | {message} | {extra}")
# import logbook
# logbook.StreamHandler(sys.stdout,
#	format_string='{record.time:%Y-%m-%d %H:%M:%S.%f} {record.level_name} {record.filename}:{record.lineno}: {record.message}').push_application()
# logger = logbook.Logger(__file__)
# logger.level = logbook.INFO

from tool_pyocr import PathSet, MyOcr, MonthDay, APP_TYPE_TO_STEM_END
from app_type import AppType

from txt_lines import TTxtLines, MTxtLines
from checksum import calculate_checksum

class Main:
    import txt_lines_db
    def __init__(self, app_type=AppType.NUL, db_fullpath: Path|Callable[[], Path]=txt_lines_db.sqlite_fullpath, my_ocr:None|MyOcr=None, tbl_ver=1):  

        if callable(db_fullpath):
            db_fullpath = db_fullpath()
        self.my_ocr = my_ocr or MyOcr()
        self.app = app_type
        self.conn = Main.txt_lines_db.connect(db_fullpath=db_fullpath)
        self.tbl_name = Main.txt_lines_db.get_table_name(self.my_ocr.date.month, version=tbl_ver)
        Main.txt_lines_db.create_tbl_if_not_exists(self.tbl_name, version=tbl_ver)
    
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
        qry = f"SELECT stem FROM `{self.tbl_name}` WHERE stem IS NOT NULL ORDER BY stem"
        with closing(self.conn.cursor()) as cur:
            cur.execute(qry)
            all = cur.fetchall()
            if all:
                return [a[0] for a in all]
    def get_having_stem(self):
        self.conn.row_factory = sqlite3.Row
        qry = f"SELECT * FROM `{self.tbl_name}` WHERE stem IS NOT NULL ORDER BY stem"
        cur = self.conn.cursor()
        cur.execute(qry)
        self.conn.row_factory = None
        return cur.fetchall()
    def get_having_pkl(self):
        pass

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
        parent = self.my_ocr.input_dir
        for img_file in self.img_dir.glob(glob_patt):
            stem = img_file.stem
            path_set = PathSet(parent, stem, ext)
            result = self.my_ocr.run_ocr(path_set=path_set, delim='')
            match result:
                case Success(value):
                    txt_lines = value # result.unwrap()
                case Failure(_): # if not is_successful(result): # type: ignore
                    raise ValueError(f"Failed to run OCR!")#Unable to extract from {path_set}")
            result = self.my_ocr.get_daAPP_TYPE_TO_STEM_ENDte(app_type=app_type, txt_lines=txt_lines)
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
            n_date = self.my_ocr.get_date(app_type=app_type, txt_lines=txt_lines)
            hrs = None
            match len(n_date):
                case 2:
                    n, date = n_date
                case 3:
                    n, date, hrs = n_date
            existing_day_list = existing_day_dict[app_type] if (existing_day_dict and app_type in existing_day_dict) else None
            if existing_day_list:
                if date.day in existing_day_list:
                    print(f"Day {date.day} of App {app_type} exists.")
            else:
                wages = self.my_ocr.get_wages(app_type=app_type, txt_lines=txt_lines)
                title = self.my_ocr.get_title(app_type, txt_lines, n)
                pkl_file_dir = self.img_dir.parent / 'pkl'
                pkl_file_dir.mkdir(parents=True, exist_ok=True)
                pkl_file_fullpath = pkl_file_dir / (stem + '.pkl')
                with pkl_file_fullpath.open('wb') as wf:
                    pkl = pickle.dump(txt_lines, wf)
                with closing(self.conn.cursor()) as cur:
                    cur.execute(f"INSERT INTO `{self.tbl_name}` VALUES (?, ?, ?, ?, ?, ?);", (app_type.value, date.day, wages, title, stem, None))
                for line in txt_lines:
                    pp(line.content)
                self.conn.commit()
                ocr_done.append((app_type, date))
        return ocr_done
    def add_image_file_without_content_into_db(self, app_type: AppType, stem: str, date: MonthDay, wages=None, title=None, pkl=None):
        with closing(self.conn.cursor()) as cur:
            cur.execute(f"INSERT INTO `{self.tbl_name}` VALUES (?, ?, ?, ?, ?, ?);", (app_type.value, date.day, wages, title, stem, pkl))
        self.conn.commit()
    def ocr_result_into_db(self, app_type_list: list[AppType]|None=None, limit=62, test=False, tbl_ver=1):
        match tbl_ver:
            case 0:
                return self.ocr_result_into_db0(app_type_list=app_type_list, limit=limit, test=test)
            case 1: 
                return self.ocr_result_into_db1(app_type_list=app_type_list, limit=limit, test=test)
            case _:
                raise ValueError(f"Unsupported tbl_ver: {tbl_ver}!")

    def ocr_result_into_db1(self, app_type_list: list[AppType]|None=None, limit=62, test=False):
        if not app_type_list:
            app_type_list = [e for e in list(AppType) if e != AppType.NUL]  
        elif isinstance(app_type_list, AppType):
            app_type_list = [app_type_list]
        assert limit > 0 
        count = 0
        with closing(self.conn.cursor()) as cur:    
            for app_type in app_type_list:
                stem_end_patt = APP_TYPE_TO_STEM_END[app_type]
                glob_patt = "*" + stem_end_patt + '.png'
                for file in self.my_ocr.input_dir.glob(glob_patt):
                    if (ocred_files:=self.get_OCRed_files()):
                        if file.stem in ocred_files:
                            continue
                    app = app_type.value
                    path_set = PathSet(self.my_ocr.input_dir, file.stem, ext=self.my_ocr.input_ext)
                    result = self.my_ocr.run_ocr(path_set)
                    match result:
                        case Success(value):
                            txt_lines = value # result.unwrap()
                        case Failure(_):
                            raise ValueError("Failed to run OCR!", path_set)
                    match(app_type):
                        case AppType.M:
                            ttxt_lines = MTxtLines(txt_lines, path_set, self.my_ocr)
                        case AppType.T:
                            ttxt_lines = TTxtLines(txt_lines, path_set, self.my_ocr)
                        case _:
                            raise NotImplementedError(f"Undefined AppType: {app_type}!")
                    date = ttxt_lines.get_date() # path_set, self.my_ocr)
                    if date.month != self.month:
                        raise ValueError(f"{date.month=} != {self.month=} in {path_set}")
                    wages = ttxt_lines.wages()
                    title = ttxt_lines.title()
                    sql = f"INSERT INTO `{self.tbl_name}` VALUES (?, ?, ?, ?, ?, ?, ?)"
                    pkl = pickle.dumps(txt_lines)
                    from checksum import calculate_checksum
                    chksum = calculate_checksum(file)
                    prm = (app_type.value, date.day, wages, title, file.stem, pkl, chksum)
                    cur.execute(sql, prm)
                    self.conn.commit()
                    logger.info("Saved into DB:  app:{}, day:{}, wages:{}, title:{}, stem:{}", *prm[:-2])
                            


    def ocr_result_into_db0(self, app_type_list: list[AppType]|None=None, limit=62, test=False):
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
                            box_pos = list(box_pos[0] + box_pos[1])
                            box_pos[0] += box_pos[3] - box_pos[1] # remove leading emoji that is about the same width of the box height
                            box_img = self.my_ocr.image.crop(tuple(box_pos)) if self.my_ocr.image else None
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
                                logger.info("Failed to get date from {} in lang=jpn+eng, try to run_ocr in lang=eng", file)
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
                                        logger.debug("date-part OCR result: {}", value)
                                        no_spc_value = value.replace(' ', '')
                                        mt = re.match(r"(\d+)月(\d+)日", no_spc_value)
                                        if mt:
                                            month, day = mt.groups()
                                            box_date = MonthDay(int(month), int(day))
                                            assert box_date.month == date.month, f"Month from box image: {box_date.month} does not match: {date.month}"
                                            if box_date.day != date.day:
                                                logger.info("Date from box image: {} is different from DB ", box_date.day)
                                                logger.info("Proceeding to fix the date in DB: {} to {}", date.day, box_date.day)
                                                result = self.fix_day(old_day=date.day, new_day=box_date.day, app=app)
                                                match result:
                                                    case Success(_):
                                                        logger.info("Fixed day from {} to {} for app {}", date.day, box_date.day, app)
                                                    case Failure(_):
                                                        logger.error("Failed to fix day from {} to {} for app {}", date.day, box_date.day, app)
                                                        raise ValueError(f"Failed to fix day from {date.day} to {box_date.day} for app {app}")
                                                logger.info("Date from box image: {} replaces the date here: {}.", box_date, date)
                                                date = box_date

                                            else:
                                                logger.error("Date from box image does not match: {}, expected: {}", box_date, date)
                                                raise ValueError(f"Date from box image does not match: {box_date}, expected: {date}")
                                        else:
                                            logger.error("Failed to get date from value: {}", no_spc_value)
                                            raise ValueError(f"Failed to get date from value: {no_spc_value}")
                                    case Failure(_):
                                        logger.error("Failed to run OCR on box image: {}", box_pos)
                                        raise ValueError(f"Failed to run OCR on box image: {box_pos}")


                                if test:
                                    debug_dir = self.my_ocr.input_dir.parent / 'DEBUG'
                                    debug_dir.mkdir(exist_ok=True)
                                    debug_fullpath = debug_dir / (file.stem + f'({box_pos}).png') 
                                    data_image.save(debug_fullpath)
                                    logger.info("Saved debug image: {}", debug_fullpath)
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

    @safe
    def fix_day(self, old_day: int, new_day: int, app: int):
        """Fix the day of the record in the DB."""
        assert app in [v for v in AppType.__members__.values() if v != AppType.NUL], f"Invalid app type: {app}"
        if old_day == new_day:
            return
        with closing(self.conn.cursor()) as cur:
            backup_sql = f"SELECT wages, title, stem FROM `{self.tbl_name}` WHERE day = {old_day} AND app = {app}"
            for old_one in cur.execute(backup_sql):
                break
            if not old_one:
                logger.error("No record found for day {} and app {}.", old_day, app)
                raise ValueError(f"No record found for day {old_day} and app {app}.")
            assert old_one, f"No record found for day {old_day} and app {app}."
            wages, title, stem = old_one[:]
            delete_sql = f"DELETE FROM `{self.tbl_name}` WHERE day = {old_day} AND app = {app}"
            cur.execute(delete_sql)
            insert_sql = f"""INSERT INTO `{self.tbl_name}`
                            (`day`, `app`, `wages`, `title`, `stem`, `txt_lines`)
                 VALUES     ({new_day}, {app}, {wages}, {title}, {stem}, NULL)
            """
            cur.execute(insert_sql)
            logger.info("Fixed day from {} to {} for app {}:keeping wages={}, title={}, stem={}", old_day, new_day, app, wages, title, stem)       

    def get_pkl_path(self, stem: str):
        """Get the path of the pkl file."""
        pkl_dir = self.img_dir.parent / 'pkl'
        pkl_dir.mkdir(parents=True, exist_ok=True)
        return pkl_dir / (stem + '.pkl')
    def get_replaced_stem_file(self, file: Path, new_stem: str):
        """Replace the stem of the file with the new stem."""
        return file.parent / (new_stem + file.suffix)

    def save_as_csv(self, dir='csv', ext='.csv', index=False):
        """Save the DB to a CSV file."""
        sql = f"SELECT app, day, wages, title, stem, checksum FROM `{self.tbl_name}` ORDER BY day, app"
        db_df = pandas.read_sql_query(sql, self.conn)
        if dir:
            output_path = self.img_dir.parent / dir
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.img_dir.parent
        output_fullpath = output_path / (self.tbl_name + ext)
        db_df.to_csv(str(output_fullpath), index=index)
        logger.info("Wrote DB `{}` to file '{}'") #, db=self.tbl_name, file=str(output_fullpath))

    def convert_to_html(self):
        """Convert the DB into HTML format."""
        sql = f"SELECT app, day, wages, title, stem, checksum FROM `{self.tbl_name}` ORDER BY day, app"
        db_df = pandas.read_sql_query(sql, self.conn)
        return db_df.to_html(index=False)

    def check_DB_T0(self, month: int, day_check_only=False):
        """Check the DB for the given month of AppType.T"""
        ocred_file_db = self.get_having_stem()
        if not ocred_file_db:
            logger.info("No OCRed files found in the DB for month: %s", month)
            return
        pkl_file_dir = self.img_dir.parent / 'pkl'
        if not pkl_file_dir.exists():
            logger.warning("pkl file directory does not exist: %s", pkl_file_dir)
            return
        pkl_files = [f for f in pkl_file_dir.iterdir() if f.is_file()]
        if not pkl_files:
            logger.warning("No pkl files found in the directory: %s", pkl_file_dir)
            return
        pkl_files_set = set([f.stem for f in pkl_files])
        ocred_file_db_set = set([f['stem'] for f in ocred_file_db])
        if pkl_files_set != ocred_file_db_set:
            logger.warning("pkl files and DB stem files do not match!")
            only_in_pkl_files = pkl_files_set - ocred_file_db_set
            if only_in_pkl_files:
                logger.info("Only in pkl files: {}", only_in_pkl_files)
            only_in_DB_stem_files = ocred_file_db_set - pkl_files_set
            if only_in_DB_stem_files:
                logger.info("Only in DB stem files: {}", only_in_DB_stem_files)
        ocred_stems = sorted(ocred_file_db_set - only_in_pkl_files)
        for stem in ocred_stems:
            pkl_fullpath = pkl_file_dir / (stem + '.pkl')
            if not pkl_fullpath.exists():
                logger.warning("pkl file does not exist: %s", pkl_fullpath)
                continue
            with pkl_fullpath.open('rb') as rf:
                txt_lines = pickle.load(rf)
            if not txt_lines:
                logger.warning("txt_lines is empty for stem: %s", stem)
                continue
            with closing(self.conn.cursor()) as cur:
                sql = f"SELECT stem, day FROM `{self.tbl_name}` WHERE app = {AppType.T.value}"
                cur.execute(sql)
                rows = cur.fetchall()
                for row in rows:
                    db_stem = row[0]
                    db_day = row[1]
                    if not db_stem:
                        raise ValueError(f"DB stem is None for day {db_day} in month {month}")
                    db_pkl_fullpath = pkl_file_dir / (db_stem + '.pkl')
                    if not db_pkl_fullpath.exists():
                        logger.error("DB pkl file does not exist: {}", db_pkl_fullpath)
                        continue
                    txt_lines = pickle.load(db_pkl_fullpath.open('rb'))
                    self.my_ocr.txt_lines = txt_lines
                    for n, txt_line in enumerate(txt_lines):
                        if txt_line.content.replace(' ', '').startswith('業務開始'):
                            break
                    if n >= len(txt_lines) - 1:
                        logger.warning("No date found in txt_lines for stem: {}", stem)
                        continue
                    date_position = txt_lines[n + 1].position
                    date_position = date_position[0] + date_position[1]
                    img_fullpath = self.img_dir / (stem + '.png') 
                    if img_fullpath.exists():
                        date_image = Image.open(img_fullpath).crop(date_position)
                        date_image_dir = self.img_dir.parent / 'DEBUG'
                        date_image_dir.mkdir(parents=True, exist_ok=True)
                        date_image_fullpath = date_image_dir / f'{stem}-{db_day:02}.png'

    def check_DB_T(self, month: int, day_check_only=False):
        """Check the DB for the given month of AppType.T"""
        ocred_file_db = self.get_having_stem()
        if not ocred_file_db:
            logger.info("No OCRed files found in the DB for month: %s", month)
            return
        pkl_file_dir = self.img_dir.parent / 'pkl'
        if not pkl_file_dir.exists():
            logger.warning("pkl file directory does not exist: %s", pkl_file_dir)
            return
        pkl_files = [f for f in pkl_file_dir.iterdir() if f.is_file()]
        if not pkl_files:
            logger.warning("No pkl files found in the directory: %s", pkl_file_dir)
            return
        pkl_files_set = set([f.stem for f in pkl_files])
        ocred_file_db_set = set([f['stem'] for f in ocred_file_db])
        if pkl_files_set != ocred_file_db_set:
            logger.warning("pkl files and DB stem files do not match!")
            only_in_pkl_files = pkl_files_set - ocred_file_db_set
            if only_in_pkl_files:
                logger.info("Only in pkl files: {}", only_in_pkl_files)
            only_in_DB_stem_files = ocred_file_db_set - pkl_files_set
            if only_in_DB_stem_files:
                logger.info("Only in DB stem files: {}", only_in_DB_stem_files)
        ocred_stems = sorted(ocred_file_db_set - only_in_pkl_files)
        for stem in ocred_stems:
            pkl_fullpath = pkl_file_dir / (stem + '.pkl')
            if not pkl_fullpath.exists():
                logger.warning("pkl file does not exist: %s", pkl_fullpath)
                continue
            with pkl_fullpath.open('rb') as rf:
                txt_lines = pickle.load(rf)
            if not txt_lines:
                logger.warning("txt_lines is empty for stem: %s", stem)
                continue
            with closing(self.conn.cursor()) as cur:
                sql = f"SELECT stem, day FROM `{self.tbl_name}` WHERE app = {AppType.T.value}"
                cur.execute(sql)
                rows = cur.fetchall()
                for row in rows:
                    db_stem = row[0]
                    db_day = row[1]
                    if not db_stem:
                        raise ValueError(f"DB stem is None for day {db_day} in month {month}")
                    db_pkl_fullpath = pkl_file_dir / (db_stem + '.pkl')
                    if not db_pkl_fullpath.exists():
                        logger.error("DB pkl file does not exist: {}", db_pkl_fullpath)
                        continue
                    txt_lines = pickle.load(db_pkl_fullpath.open('rb'))
                    self.my_ocr.txt_lines = txt_lines
                    for n, txt_line in enumerate(txt_lines):
                        if txt_line.content.replace(' ', '').startswith('業務開始'):
                            break
                    if n >= len(txt_lines) - 1:
                        logger.warning("No date found in txt_lines for stem: {}", stem)
                        continue
                    date_position = txt_lines[n + 1].position
                    date_position = date_position[0] + date_position[1]
                    img_fullpath = self.img_dir / (stem + '.png') 
                    if img_fullpath.exists():
                        date_image = Image.open(img_fullpath).crop(date_position)
                        date_image_dir = self.img_dir.parent / 'DEBUG'
                        date_image_dir.mkdir(parents=True, exist_ok=True)
                        date_image_fullpath = date_image_dir / f'{stem}-{db_day:02}.png'
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
                                    if date.day != db_day:
                                        logger.warning("Date from OCRed image: {} does not match DB day: {}", date, db_day)
                                        if day_check_only:
                                            print(f"{db_day:02}:OCRed date {date.day} does not match DB day {db_day}")
                                        else:
                                            yn = input(f"Replace DB day {db_day} with OCRed date {date.day}? (y/n): ")
                                            if yn.lower() == 'y':
                                                self.fix_day(old_day=db_day, new_day=date.day, app=AppType.T.value)
                                                logger.info("Replaced DB day {} with OCRed date {}", db_day, date.day)
                                else:
                                    logger.error("Failed to get date of day {} from value: {}", db_day, no_spc_value)
                                    if day_check_only:
                                        print(f"{db_day:02}:failed to get date from OCRed image: {no_spc_value}")
                                    else:
                                        raise ValueError(f"Failed to get date of day {db_day} from value: {no_spc_value}")
                                    # TODO: check all date positions
                    else:
                        logger.error("Image file does not exist: {}", img_fullpath)
                        raise ValueError(f"Image file does not exist: {img_fullpath}")

    def fix_title(self, day: int, month=0, app_type=AppType.T):
        if not month:
            month = self.my_ocr.date.month
            logger.debug("month==0 is reset as:{}", month)
        with closing(self.conn.cursor()) as cur:
            sql = f"SELECT title, stem, txt_lines, checksum FROM `{self.tbl_name}` WHERE app = {app_type.value} AND day = {day}"
            cur.execute(sql)
            row = cur.fetchone()
            # if one[0] is NG, then replace the title
            yn = input(f"Is '{row[0]}' good for the title?:(y/N)")
            if not yn or yn[0].lower() == 'n':
                # load original image
                if not (stem:=row[1]):
                    raise ValueError(f"No stem for the record!")
                img_pathset = PathSet(self.img_dir, stem, '.png')
                img_fullpath = img_pathset.to_path()
                if not img_fullpath.exists():
                    raise ValueError(f"`{img_fullpath=}` does not exists!")
                db_chksum = row[3]
                if not db_chksum:
                    raise ValueError("No checksum in DB row!")
                if calculate_checksum(img_fullpath) != db_chksum:
                    raise ValueError("Checksum of the file doesn't match with DB's!")
                txt_lines_pkl = row[2]
                if not txt_lines_pkl:
                    raise ValueError(f"No txt_lines in the row in DB!")
                txt_lines = pickle.loads(txt_lines_pkl)
                if not txt_lines:
                    raise ValueError(f"No txt_lines loaded from its pickle!")
                t_txt_lines = TTxtLines(txt_lines=txt_lines, img_pathset=img_pathset, my_ocr=self.my_ocr)
                new_title = t_txt_lines.get_title()
                if not new_title:
                    raise ValueError("Unable to get a new title!")
                yn = input(f"Do you want to update to the new title?(Y/n): {new_title}")
                if not yn or yn[0].lower() == 'y':
                    # TODO: update DB sequence
                    sql = f"UPDATE `{self.tbl_name}` SET title = '{new_title}' WHERE app = {app_type.value} AND day = {day}"
                    cur.execute(sql)
                    self.conn.commit()
                    logger.info("DB is updated app:{}, day:{}, title:{}", app_type, day, new_title)


                



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
    """Check the DB for the given month of AppType.T"""
    assert app != AppType.NUL
    my_ocr = MyOcr(month=month)
    main = Main(my_ocr=my_ocr, app_type=AppType.T)
    with closing(self.conn.cursor()) as cur:
        for app_type in app_type_list:
            stem_end_patt = APP_TYPE_TO_STEM_END[app_type]
            glob_patt = "*" + stem_end_patt + '.png'
            for file in self.my_ocr.input_dir.glob(glob_patt):
                if file.stem in self.get_OCRed_files():
                    continue
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
from functools import wraps    
def run_main_func(func, month=0, app_typ=AppType.T):
  def run_main_wrapper(func):
    @wraps(func)
    def Inner(*args, **kwargs):
        def wrapper(*args, **kwargs):
            my_ocr = MyOcr(month=kwargs['month'])
            main = Main(my_ocr=my_ocr, app_type=kwargs['app_type'])
            main.func(*args, **kwargs)
        return wrapper
    return Inner
  return run_main_wrapper

class RunMain:
    def __init__(self, month=0, app_type=AppType.NUL):
            my_ocr = MyOcr(month=month)
            self.main = Main(my_ocr=my_ocr, app_type=app_type)
    def run(self):
        pass

class SaveAsCsv(RunMain):
    def __init__(self, month=0, app_type=AppType.NUL):
        super().__init__(month=month, app_type=app_type)

    @override
    def run(self):
        self.main.save_as_csv()

def run_fix_title(day, month=0):
    Main(my_ocr=MyOcr(month=month)).fix_title(day=day)

def run_save_as_csv(month: int):
    """save DB table of the month except pickle of txt_lines"""
    Main(my_ocr=MyOcr(month=month)).save_as_csv()

def run_check_DB_T(month: int, day_check_only=False):
    """Check the DB for the given month of AppType.T"""
    my_ocr = MyOcr(month=month)
    main = Main(my_ocr=my_ocr, app_type=AppType.T)
    main.check_DB_T(month=month, day_check_only=day_check_only)

def run_ocr(month: int=0, limit=62, app_type: AppType = AppType.NUL, test=False, tbl_ver=1):
    """Run OCR and save result into DB."""
    my_ocr = MyOcr(month=month)
    main = Main(my_ocr=my_ocr, app_type=app_type, tbl_ver=tbl_ver)

    main.ocr_result_into_db(limit=limit, app_type_list=app_type, test=test)

def save_as_csv(month: int=0, app_type: AppType = AppType.NUL, tbl_ver=1):
    """Run OCR and save result into DB."""
    my_ocr = MyOcr(month=month)
    main = Main(my_ocr=my_ocr, app_type=app_type, tbl_ver=tbl_ver)
    main.save_as_csv()

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
            s = self.func(**self.kwargs) #if self.kwargs else self.func()
            if isinstance(s, str):
                print(s)
def get_options(month=0):
    my_ocr = MyOcr(month=month) 
    main = Main(my_ocr=my_ocr)
    return [
        FunctionItem('Exit', None),
        FunctionItem('save OCR result into DB', main.ocr_result_into_db),
        FunctionItem('Convert DB to HTML', main.convert_to_html),
    ]

def run_main(options: Sequence[FunctionItem]|None=None, month=0):
    options = options or get_options(month=month)
    for n, option in enumerate(options):
        print(f"{n}. {option.title}")
    choice = int(input(f"Choose(0 to {len(options) - 1}):"))
    if choice:
        options[choice].exec()

if __name__ == '__main__':
    from sys import argv
    month = int(argv[1]) if len(argv) > 1 else int(input("Month for Data?:") or '0')
    run_main(month=month)
    #month = int(sys.argv[1])
    #run_save_as_csv(month=month)
    #run_fix_title(day)
    #art = sys.argv[2][0].upper()
    #app_type = {'T':AppType.T, 'M':AppType.M}[art]
    #run_save_as_csv(month)
    #run_ocr(5)
