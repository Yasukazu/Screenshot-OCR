from contextlib import closing
from enum import IntEnum
from types import MappingProxyType
import re
from typing import Sequence, Callable, Any
from collections import namedtuple
from pprint import pp
from pathlib import Path
from dataclasses import dataclass
import pandas
from dotenv import load_dotenv
load_dotenv()
from PIL import Image, ImageDraw, ImageEnhance
import pyocr
import pyocr.builders
from pyocr.builders import LineBox
from returns.pipeline import is_successful, UnwrapFailedError

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

class AppType(IntEnum):
    NUL = 0
    T = 1
    M = 2

APP_TYPE_TO_STEM_END = MappingProxyType({
    AppType.T: ".co.taimee",
    AppType.M: ".mercari.work.android"
})

class TTxtLines:
    TITLE = 1
    def __init__(self, txt_lines: Sequence[LineBox]):
        self.txt_lines = txt_lines

    def title(self, n=0):
        return self.txt_lines[1].content.replace(' ', '')

class MTxtLines(TTxtLines):
    def title(self, n: int):
        return ''.join([self.txt_lines[i].content.replace(' ', '') for i in range(n - 3, n - 1)])
class OCRError(Exception):
    pass
from returns.result import Result, Failure, Success, safe
class MyOcr:
    from path_feeder import input_dir_root, input_ext
    tools = pyocr.get_available_tools()
    tool = tools[0]
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

    M_DATE_PATT_1 = re.compile(r"[^\d]*((\d+)/(\d+))\(")
    M_DATE_PATT_2 = re.compile(r".*時間$")

    @classmethod
    def get_date(cls, app_type: AppType, txt_lines: Sequence[LineBox]):
        match app_type:
            case AppType.M:
                for n in range(len(txt_lines)):
                    cntnt = txt_lines[n].content.replace(' ', '')
                    mt = MyOcr.M_DATE_PATT_1.match(cntnt)
                    mt2 = MyOcr.M_DATE_PATT_2.match(cntnt)
                    if mt and mt2:
                        grps = mt.groups()
                        date = Date(int(grps[1]), int(grps[2]))
                        return n, date
                raise ValueError("Unmatch AppType.M txt_lines!")                
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
        from path_feeder import PathFeeder
        #self.path_feeder = PathFeeder(input_dir=MyOcr.input_dir_root, type_dir=False, month=month)
        self.txt_lines: Sequence[pyocr.builders.LineBox] | None = None
        self.image: Image.Image | None = None
    @property
    def input_dir(self):
        return MyOcr.input_dir_root / str(self.date.year) / f'{self.date.month:02}'
    '''def each_path_set(self):
        for stem in self.path_feeder.feed(delim=self.delim, padding=False):
            yield PathSet(self.path_feeder.dir, stem, self.path_feeder.ext)'''
    @safe
    def run_ocr(self, path_set: PathSet, lang='jpn+eng', delim='',
                builder=pyocr.builders.LineBoxBuilder(tesseract_layout=3))-> Sequence[pyocr.builders.LineBox]:#, Exception]: # Digit
        #stem_without_delim = ''.join([s for s in path_set[1] if s!= self.delim])
        fullpath = path_set.path / (path_set.stem_without_delim(delim) + path_set.ext)
        img = Image.open(fullpath).convert('L')
        enhancer= ImageEnhance.Contrast(img)
        self.image = enhancer.enhance(2.0)
        txt_lines = self.tool.image_to_string(
            self.image,
            lang=lang,
            builder=builder
        )
        if txt_lines:
            assert isinstance(txt_lines[0], pyocr.builders.LineBox)
        self.txt_lines = txt_lines
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

    '''def check_date(self, path_set: PathSet):#, txt_lines: Sequence[pyocr.builders.LineBox]):
        if not self.txt_lines:
            raise ValueError('txt_lines is None!')
        n_gyoumu = next_gyoumu(self.txt_lines)
        gyoumu_date = get_date(self.txt_lines[n_gyoumu])
        if gyoumu_date != (int(path_set.stem.split()[0]), int(path_set.stem.split()[1])): # path_feeder.month, 
            raise ValueError(f"Unmatch {gyoumu_date} : {path_set.stem}!")'''

import pickle

class Main:
    import txt_lines_db
    def __init__(self, month=0, app=AppType.NUL, year=0):
        self.my_ocr = MyOcr(month=month, year=year)
        #self.img_dir = self.my_ocr.input_dir
        #month = self.my_ocr.month
        #img_parent_dir = self.img_dir.parent
        self.app = app # tm
        # txt_lines_db = TxtLinesDB(img_parent_dir=img_parent_dir)
        self.conn = Main.txt_lines_db.connect()
        self.tbl_name = Main.txt_lines_db.get_table_name(month)
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
        return self.my_ocr.month

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

    def get_existing_days(self, app_type=AppType.NUL):
        if not app_type.value:
            from collections import defaultdict
            qry = f"SELECT app, day, stem from `{self.tbl_name}` order by app, day;"
            with closing(self.conn.cursor()) as cur:
                cur.execute(qry)
                all = cur.fetchall()
                if all:
                    r_dict = defaultdict(dict)
                    for app, day, stem in all:
                        app_type = AppType(app)
                        r_dict[app_type][day] = stem
                    return r_dict
                else:
                    return
        qry = f"SELECT day from `{self.tbl_name}` where app = ?;"
        prm = (app_type.value, ) # date.day)
        with closing(self.conn.cursor()) as cur:
            result = cur.execute(qry, prm)
            if result:
                return [r[0]    for r in result]
    def add_image_files_as_txt_lines(self, app_type: AppType, ext='.png'):
        if app_type == AppType.NUL:
            raise ValueError('AppType is NUL!')
        ocr_done = []
        for img_file in self.img_dir.glob('*' + ext):
            if not img_file.stem.endswith(APP_TYPE_TO_STEM_END[app_type]):
                continue
            #ext_dot = img_file.name.rfind('.')
            #ext = img_file.name[ext_dot:]
            stem = img_file.stem
            parent = self.my_ocr.input_dir
            path_set = PathSet(parent, stem, ext)
            result = self.my_ocr.run_ocr(path_set=path_set, delim='')
            if not is_successful(result): # type: ignore
                raise ValueError(f"Failed to run OCR!")#Unable to extract from {path_set}")
            
                txt_lines = result.unwrap()
            # app_type = self.my_ocr.get_app_type(txt_lines) if txt_lines else AppType.NUL
            n, date = self.my_ocr.get_date(app_type=app_type, txt_lines=txt_lines)
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
                    pp(line.content)
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
            try:#if not is_successful(result): # type: ignore
                txt_lines = result.unwrap()
            except UnwrapFailedError as uwerr:
                raise ValueError("Failed to run OCR!", path_set) from uwerr
            n, date = self.my_ocr.get_date(app_type=app_type, txt_lines=txt_lines)
            existing_day_list = existing_day_dict[app_type]
            if existing_day_list and (date.day in existing_day_list):
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
    def ocr_result_into_db(self):
        t_patt = APP_TYPE_TO_STEM_END[AppType.T]
        m_patt = APP_TYPE_TO_STEM_END[AppType.M]
        with closing(self.conn.cursor()) as cur:
            for file in self.my_ocr.input_dir.iterdir():
                if not (file.is_file() and file.suffix == '.png'):
                    continue
                app_type = AppType.NUL
                if file.stem.endswith(t_patt):
                    app_type = AppType.T
                elif file.stem.endswith(m_patt):
                    app_type = AppType.M
                if app_type == AppType.NUL:
                    raise ValueError("Not supported file!")
                app = app_type.value
                path_set = PathSet(self.my_ocr.input_dir, file.stem, ext=self.my_ocr.input_ext)
                result = self.my_ocr.run_ocr(path_set)
                #if not is_successful(result): # type: ignore raise ValueError("OCR failed!")
                txt_lines = result.unwrap()
                n, date = self.my_ocr.get_date(app_type=app_type, txt_lines=txt_lines)
                exists_sql = f"SELECT day, app FROM `{self.tbl_name}` WHERE day = {date.day} AND app = {app};"
                cur.execute(exists_sql)
                one = cur.fetchone()
                if not one:
                    tm_txt_lines = TTxtLines(txt_lines) if app_type == AppType.T else MTxtLines(txt_lines)
                    title = tm_txt_lines.title(n)
                    pkl = pickle.dumps(txt_lines)
                    cur.execute(f"INSERT INTO `{self.tbl_name}` VALUES (?, ?, ?, ?, ?, ?);", (app, date.day, None, title, file.stem, pkl))
        self.conn.commit()
    def save_as_csv(self):
        #conn = sqlite3.connect(db_file, isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES)
        table = f"text_lines-{self.my_ocr.date.month:02}"
        sql = f"SELECT * FROM `{table}`"
        db_df = pandas.read_sql_query(sql, self.conn)
        output_path = self.img_dir# / str(self.my_ocr.date.year) / f"{self.my_ocr.date.month:02}"
        breakpoint()
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


import click
@click.group()
def cli():
    pass
@cli.command()
@click.argument('month')
def run_ocr(month: str):
    m = int(month)
    main = Main(m)
    main.ocr_result_into_db()
class FunctionItem:
    def __init__(self, title: str, func: Callable | None, kwargs: dict[str, Any]={}):
        self.title = title
        self.func = func
        self.kwargs = kwargs
    def exec(self):
        if self.func:
            self.func(**self.kwargs) #if self.kwargs else self.func()
def get_options(month=0):
    main = Main(month)
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
    app = AppType.T if int(sys.argv[1]) == 1 else AppType.M
    month = int(sys.argv[2])
    main = Main(month=month, app=app)
    if len(sys.argv) > 4:
        day = int(sys.argv[3])
        stem = sys.argv[4]
        match app:
            case 1:
                app_type = AppType.T
            case 2:
                app_type = AppType.M
            case _:
                raise ValueError('Unsupported app type!')
        main.add_image_file_without_content_into_db(app_type, stem, Date(month=month, day=day))
        sys.exit(0)
    from consolemenu import ConsoleMenu, SelectionMenu
    from consolemenu.items import FunctionItem, SubmenuItem
    import consolemenu.items
    def show_total_wages():
        print(main.sum_wages())
        input('Hit Enter key, please:')
    def show_existing_days():
        app = input("App type: Tm: 1, MH: 2")
        app_type = [AppType.NUL, AppType.T, AppType.M][int(app)]
        print(main.get_existing_days(app_type ))
        input('Hit Enter key, please:')
    def run_ocr():
        app_type_list = [AppType.T, AppType.M]
        patt_list = [APP_TYPE_TO_STEM_END[AppType.T], APP_TYPE_TO_STEM_END[AppType.M]]#["t.*.png", "Screenshot_*.png"]
        selected = SelectionMenu.get_selection([APP_TYPE_TO_STEM_END[at] for at in app_type_list])
        txt_lines = main.add_image_files_as_txt_lines(
            app_type_list[selected]
        )
        txt_lines_len = len(txt_lines) if txt_lines else 0
        input(f'{txt_lines_len} file(s) is / are OCRed. Hit Enter key to return to the main menu:')
    function_items = [
        FunctionItem("Show the total wages", show_total_wages),
        FunctionItem("Show existing days", show_existing_days),# ["Enter"]),
    ]
    menu = ConsoleMenu("Menu", "-- OCR DB --")
    for f_it in function_items:
        menu.append_item(f_it)
    submenu = ConsoleMenu("SubMenu", "-- Run OCR --")
    submenu.append_item(FunctionItem("Run OCR with file name patterns:", run_ocr))
    submenu_item = SubmenuItem("SubMenu", submenu=submenu, menu=menu)
    menu.append_item(submenu_item)
    
    menu.show()
